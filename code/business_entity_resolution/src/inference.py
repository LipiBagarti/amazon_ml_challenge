"""
Full Inference Pipeline for Business Entity Resolution (Amazon ML Challenge 2026).
Processes test data partitioned by country (France, US, India) to maintain
minimal memory footprint (< 2 GB RAM) and maximum throughput.
Generates:
1. output/candidate_pairs.tsv
2. output/matching_results.tsv
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl

sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from normalization import (
    normalize_name, normalize_address, extract_numbers, extract_postal_code
)
from blocking import BlockingIndex, clean_num_token
from features import compute_pairwise_features
from model import EntityResolutionModel
from decision import apply_decision_rules


def train_production_model(train_sample_size: int = 15000) -> EntityResolutionModel:
    """Train production LightGBM model on a diverse sample from training data."""
    print("=== Training Production Matching Model ===")
    t0 = time.time()
    
    s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=train_sample_size)
    s1_ids = set(s1_df["entity_id"].to_list())
    
    gt_df = pl.scan_csv("dataset/train/train_ground_truth.tsv", separator="\t").filter(
        pl.col("source1_entity_id").is_in(list(s1_ids))
    ).collect()
    
    gt_map = {}
    true_target_ids = set()
    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        t_str = r["matched_entity_ids"]
        if t_str and str(t_str).strip() and str(t_str).lower() != "nan":
            targets = set(t_str.split(","))
            gt_map[sid] = targets
            true_target_ids.update(targets)
        else:
            gt_map[sid] = set()
            
    t_s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(list(true_target_ids))
    ).collect()
    t_s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(list(true_target_ids))
    ).collect()
    bg_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=25000)
    bg_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=25000)
    targets_df = pl.concat([t_s2, t_s3, bg_s2, bg_s3]).unique(subset=["entity_id"])
    
    from dataset_builder import build_labeled_dataset
    X_train, y_train, _, _ = build_labeled_dataset(
        s1_df, targets_df, gt_map, max_candidates_per_entity=30, min_similarity_floor=24.0, verbose=False
    )
    
    model = EntityResolutionModel(
        num_leaves=41,
        learning_rate=0.08,
        n_estimators=160,
        min_child_samples=25,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Hard negative mining
    sample_weights = model.mine_hard_negatives(X_train, y_train, weight_multiplier=2.5)
    model.fit(X_train, y_train, sample_weights=sample_weights)
    
    print(f"Production model trained on {len(X_train):,} pairs in {time.time() - t0:.2f}s.")
    return model


def run_country_inference(
    country: str,
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    model: EntityResolutionModel,
    threshold: float = 0.68,
    max_cands_per_entity: int = 30,
    min_similarity_floor: float = 24.0,
    batch_size: int = 25000
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Run candidate generation, feature extraction, and matching for one country partition.
    Returns: (matching_results_dict, candidate_pairs_dict)
    """
    print(f"\n--- Running Inference for Country: {country} ---")
    print(f"S1 count: {len(s1_df):,}, S2 count: {len(s2_df):,}, S3 count: {len(s3_df):,}")
    
    # 1. Index S2 and S3 for this country
    t0 = time.time()
    targets_df = pl.concat([s2_df, s3_df])
    
    index = BlockingIndex()
    target_clean_dict = {}
    
    for row in targets_df.iter_rows(named=True):
        tid = row["entity_id"]
        c_name = normalize_name(row["business_name"])
        compact_n = c_name.replace(" ", "")
        raw_a = row["business_address"]
        c_addr = normalize_address(raw_a) if raw_a and str(raw_a).lower() != "nan" else ""
        raw_nums = extract_numbers(c_addr)
        nums = [clean_num_token(n) for n in raw_nums]
        postal = extract_postal_code(c_addr)
        c_val = country
        
        target_clean_dict[tid] = (c_name, compact_n, c_addr, nums, postal, c_val)
        index.add_target(tid, row["business_name"], row["business_address"], country)
        
    print(f"[{country}] Target index built for {len(targets_df):,} targets in {time.time() - t0:.2f}s.")
    
    # 2. Process S1 entities in batches
    matching_dict = {}
    candidate_dict = {}
    
    all_pairs_s1 = []
    all_pairs_target = []
    all_pair_features = []
    
    s1_rows = s1_df.to_dicts()
    t_start = time.time()
    
    for idx, s1_r in enumerate(s1_rows):
        s1_id = s1_r["entity_id"]
        s1_cname = normalize_name(s1_r["business_name"])
        s1_compact = s1_cname.replace(" ", "")
        s1_raw_a = s1_r["business_address"]
        s1_caddr = normalize_address(s1_raw_a) if s1_raw_a and str(s1_raw_a).lower() != "nan" else ""
        s1_raw_nums = extract_numbers(s1_caddr)
        s1_nums = [clean_num_token(n) for n in s1_raw_nums]
        s1_postal = extract_postal_code(s1_caddr)
        
        cands = index.get_candidates(
            s1_id=s1_id,
            name=s1_r["business_name"],
            addr=s1_r["business_address"],
            country=country,
            max_candidates=max_cands_per_entity,
            min_similarity_floor=min_similarity_floor
        )
        
        cand_ids = [c[0] for c in cands]
        candidate_dict[s1_id] = cand_ids
        
        if not cands:
            continue
            
        top_score = cands[0][1]
        total_cands = len(cands)
        
        for rank, (cand_id, coarse_score) in enumerate(cands, start=1):
            if cand_id not in target_clean_dict:
                continue
            t_data = target_clean_dict[cand_id]
            feat_vec = compute_pairwise_features(
                s1_clean_name=s1_cname,
                s1_compact_name=s1_compact,
                s1_clean_addr=s1_caddr,
                s1_nums=s1_nums,
                s1_postal=s1_postal,
                s1_country=country,
                target_id=cand_id,
                t_clean_name=t_data[0],
                t_compact_name=t_data[1],
                t_clean_addr=t_data[2],
                t_nums=t_data[3],
                t_postal=t_data[4],
                t_country=country,
                coarse_score=coarse_score,
                cand_rank=rank,
                top_score=top_score,
                total_cands=total_cands
            )
            all_pairs_s1.append(s1_id)
            all_pairs_target.append(cand_id)
            all_pair_features.append(feat_vec)
            
        if (idx + 1) % 50000 == 0 or (idx + 1) == len(s1_rows):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            print(f"[{country}] Processed {idx+1:,}/{len(s1_rows):,} S1 queries ({rate:,.0f} queries/sec)...")
            
    print(f"[{country}] Total candidate pairs scored: {len(all_pairs_s1):,}")
    
    # 3. Model scoring and Global 1-to-1 Resolution
    if all_pair_features:
        X_mat = np.array(all_pair_features, dtype=np.float32)
        probs = model.predict_proba(X_mat)
        
        matches = apply_decision_rules(
            pair_s1_ids=all_pairs_s1,
            pair_target_ids=all_pairs_target,
            probabilities=probs,
            threshold=threshold,
            enforce_one_to_one=True
        )
        
        # Populate matching_dict for all S1 entities (empty list for singletons)
        for s1_r in s1_rows:
            sid = s1_r["entity_id"]
            if sid in matches:
                matching_dict[sid] = sorted(list(matches[sid]))
            else:
                matching_dict[sid] = []
    else:
        for s1_r in s1_rows:
            matching_dict[s1_r["entity_id"]] = []
            
    return matching_dict, candidate_dict


def run_full_inference(
    output_dir: str = "output",
    threshold: float = 0.68
):
    """Run full test inference and write submission files."""
    os.makedirs(output_dir, exist_ok=True)
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # 1. Train model
    model = train_production_model(train_sample_size=15000)
    
    # 2. Iterate through countries (France, US, India)
    countries = ["France", "US", "India"]
    
    all_matching = {}
    all_candidates = {}
    
    for country in countries:
        print(f"\nLoading Test Data for {country}...")
        # Scan and filter per country using Polars
        s1 = pl.scan_csv("dataset/test/test_source1.tsv", separator="\t").filter(pl.col("country") == country).collect()
        s2 = pl.scan_csv("dataset/test/test_source2.tsv", separator="\t").filter(pl.col("country") == country).collect()
        s3 = pl.scan_csv("dataset/test/test_source3.tsv", separator="\t").filter(pl.col("country") == country).collect()
        
        c_matches, c_cands = run_country_inference(
            country=country,
            s1_df=s1,
            s2_df=s2,
            s3_df=s3,
            model=model,
            threshold=threshold
        )
        
        all_matching.update(c_matches)
        all_candidates.update(c_cands)
        
        # Free memory
        del s1, s2, s3, c_matches, c_cands
        
    # 3. Write final outputs preserving exact test_source1 entity order
    print(f"\nWriting final outputs to {output_dir}/...")
    s1_all_ids = pl.scan_csv("dataset/test/test_source1.tsv", separator="\t").select("entity_id").collect()["entity_id"].to_list()
    
    print(f"Total S1 entities in test: {len(s1_all_ids):,}")
    assert len(all_matching) == len(s1_all_ids), f"Mismatch in matching count: {len(all_matching)} vs {len(s1_all_ids)}"
    assert len(all_candidates) == len(s1_all_ids), f"Mismatch in candidate count: {len(all_candidates)} vs {len(s1_all_ids)}"
    
    # Write matching_results.tsv
    with open(matching_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            matches = all_matching.get(sid, [])
            match_str = ",".join(matches)
            f.write(f"{sid}\t{match_str}\n")
            
    # Write candidate_pairs.tsv
    with open(candidate_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_all_ids:
            cands = all_candidates.get(sid, [])
            cand_str = ",".join(cands)
            f.write(f"{sid}\t{cand_str}\n")
            
    print(f"Generated {matching_tsv} ({os.path.getsize(matching_tsv)/(1024*1024):.2f} MB)")
    print(f"Generated {candidate_tsv} ({os.path.getsize(candidate_tsv)/(1024*1024):.2f} MB)")
    print("=== Inference Completed Successfully ===")


if __name__ == "__main__":
    run_full_inference()
