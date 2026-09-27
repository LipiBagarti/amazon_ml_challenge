import sys
import os
import time
from collections import Counter
import polars as pl
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from normalization import normalize_name, normalize_address, extract_numbers
from blocking import BlockingIndex

# Test the fast candidate generation logic
print("=== Testing Ultra-Fast Candidate Pruning ===")
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=3000)
all_s1_ids = set(s1_df["entity_id"].to_list())

gt_df = pl.scan_csv("dataset/train/train_ground_truth.tsv", separator="\t").filter(
    pl.col("source1_entity_id").is_in(list(all_s1_ids))
).collect()

true_matches_map = {}
all_true_target_ids = set()
for r in gt_df.iter_rows(named=True):
    sid = r["source1_entity_id"]
    t_str = r["matched_entity_ids"]
    if t_str and str(t_str).strip() and str(t_str).lower() != "nan":
        targets = set(t_str.split(","))
        true_matches_map[sid] = targets
        all_true_target_ids.update(targets)

t_s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()
t_s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()
bg_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=20000)
bg_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=20000)
targets_df = pl.concat([t_s2, t_s3, bg_s2, bg_s3]).unique(subset=["entity_id"])

index = BlockingIndex()
for row in targets_df.iter_rows(named=True):
    index.add_target(
        entity_id=row["entity_id"],
        name=row["business_name"],
        addr=row["business_address"],
        country=row["country"]
    )

# Benchmark query throughput and recall
t0 = time.time()
total_true = 0
retrieved = 0

s1_eval = s1_df.filter(pl.col("entity_id").is_in(list(true_matches_map.keys()))).to_dicts()
for r in s1_eval:
    sid = r["entity_id"]
    true_t = true_matches_map[sid]
    total_true += len(true_t)
    
    cands = index.get_candidates(
        s1_id=sid,
        name=r["business_name"],
        addr=r["business_address"],
        country=r["country"],
        max_candidates=35,
        min_similarity_floor=22.0
    )
    cand_ids = {c[0] for c in cands}
    retrieved += len(cand_ids & true_t)

elapsed = time.time() - t0
rate = len(s1_eval) / elapsed
recall = retrieved / total_true * 100

print(f"Processed {len(s1_eval):,} queries in {elapsed:.2f}s ({rate:,.0f} queries/sec)!")
print(f"Recall Ceiling: {recall:.2f}% ({retrieved:,}/{total_true:,})")
