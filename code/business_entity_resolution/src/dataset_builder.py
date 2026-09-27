"""
Dataset Builder Module for Business Entity Resolution.
Generates labeled pairwise feature datasets directly from candidate blocking
on entity-level train/validation splits (zero entity leakage).
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl

sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from normalization import (
    normalize_name, normalize_address, extract_numbers, extract_postal_code
)
from blocking import BlockingIndex, clean_num_token
from features import compute_pairwise_features, FEATURE_NAMES


def build_labeled_dataset(
    s1_df: pl.DataFrame,
    targets_df: pl.DataFrame,
    gt_map: Dict[str, Set[str]],
    max_candidates_per_entity: int = 35,
    min_similarity_floor: float = 22.0,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build labeled feature matrix X and target vector y from candidate generation.
    Returns: (X, y, list_of_s1_ids, list_of_target_ids)
    """
    if verbose:
        print(f"Indexing {len(targets_df):,} target records...")
    t0 = time.time()
    
    index = BlockingIndex()
    target_clean_dict = {}  # target_id -> (clean_name, compact_name, clean_addr, nums, postal, country)
    
    for row in targets_df.iter_rows(named=True):
        tid = row["entity_id"]
        c_name = normalize_name(row["business_name"])
        compact_n = c_name.replace(" ", "")
        raw_a = row["business_address"]
        c_addr = normalize_address(raw_a) if raw_a and str(raw_a).lower() != "nan" else ""
        raw_nums = extract_numbers(c_addr)
        nums = [clean_num_token(n) for n in raw_nums]
        postal = extract_postal_code(c_addr)
        country = row["country"].strip() if row["country"] else "UNKNOWN"
        
        target_clean_dict[tid] = (c_name, compact_n, c_addr, nums, postal, country)
        index.add_target(tid, row["business_name"], row["business_address"], row["country"])
        
    if verbose:
        print(f"Target index built in {time.time() - t0:.2f}s.")
        print(f"Generating candidate pairs and extracting features for {len(s1_df):,} S1 entities...")
    t1 = time.time()
    
    features_list = []
    labels_list = []
    pair_s1_ids = []
    pair_target_ids = []
    
    s1_rows = s1_df.to_dicts()
    for s1_r in s1_rows:
        s1_id = s1_r["entity_id"]
        s1_country = s1_r["country"].strip() if s1_r["country"] else "UNKNOWN"
        s1_cname = normalize_name(s1_r["business_name"])
        s1_compact = s1_cname.replace(" ", "")
        s1_raw_a = s1_r["business_address"]
        s1_caddr = normalize_address(s1_raw_a) if s1_raw_a and str(s1_raw_a).lower() != "nan" else ""
        s1_raw_nums = extract_numbers(s1_caddr)
        s1_nums = [clean_num_token(n) for n in s1_raw_nums]
        s1_postal = extract_postal_code(s1_caddr)
        
        true_targets = gt_map.get(s1_id, set())
        
        candidates = index.get_candidates(
            s1_id=s1_id,
            name=s1_r["business_name"],
            addr=s1_r["business_address"],
            country=s1_r["country"],
            max_candidates=max_candidates_per_entity,
            min_similarity_floor=min_similarity_floor
        )
        
        if not candidates:
            continue
            
        top_score = candidates[0][1]
        total_cands = len(candidates)
        
        for rank, (cand_id, coarse_score) in enumerate(candidates, start=1):
            if cand_id not in target_clean_dict:
                continue
                
            t_data = target_clean_dict[cand_id]
            
            feat_vec = compute_pairwise_features(
                s1_clean_name=s1_cname,
                s1_compact_name=s1_compact,
                s1_clean_addr=s1_caddr,
                s1_nums=s1_nums,
                s1_postal=s1_postal,
                s1_country=s1_country,
                target_id=cand_id,
                t_clean_name=t_data[0],
                t_compact_name=t_data[1],
                t_clean_addr=t_data[2],
                t_nums=t_data[3],
                t_postal=t_data[4],
                t_country=t_data[5],
                coarse_score=coarse_score,
                cand_rank=rank,
                top_score=top_score,
                total_cands=total_cands
            )
            
            label = 1 if cand_id in true_targets else 0
            
            features_list.append(feat_vec)
            labels_list.append(label)
            pair_s1_ids.append(s1_id)
            pair_target_ids.append(cand_id)
            
    if verbose:
        print(f"Features extracted in {time.time() - t1:.2f}s.")
        print(f"Total candidate pairs: {len(features_list):,}")
        pos = sum(labels_list)
        neg = len(labels_list) - pos
        print(f"Class distribution: Positives={pos:,} ({pos/len(labels_list)*100:.2f}%), Negatives={neg:,} ({neg/len(labels_list)*100:.2f}%)")
        print(f"Imbalance ratio (Neg:Pos) = {neg/max(pos, 1):.2f}:1")
        
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y, pair_s1_ids, pair_target_ids
