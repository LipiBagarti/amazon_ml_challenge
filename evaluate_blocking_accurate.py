import sys
import os
import polars as pl
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from normalization import normalize_name, normalize_address, extract_numbers
from blocking import BlockingIndex

# Fix evaluation: sample S1 from train_source1 FIRST, then get ground truth for those exact S1 entities!
print("Reading S1 sample...")
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=5000)
s1_ids = set(s1_df["entity_id"].to_list())

print("Reading ground truth for these S1 entities...")
gt = pl.scan_csv("dataset/train/train_ground_truth.tsv", separator="\t").filter(
    pl.col("source1_entity_id").is_in(list(s1_ids))
).collect()

gt_matches = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))

true_matches_map = {}
all_true_target_ids = set()
total_true_links = 0

for row in gt_matches.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    targets = set(row["matched_entity_ids"].split(","))
    true_matches_map[s1_id] = targets
    all_true_target_ids.update(targets)
    total_true_links += len(targets)

s1_eval = s1_df.filter(pl.col("entity_id").is_in(list(true_matches_map.keys())))
print(f"Evaluated S1 entities: {len(s1_eval):,} with {total_true_links:,} true target links.")

# Fetch all true targets
print(f"Fetching {len(all_true_target_ids):,} true targets from S2 and S3...")
targets_s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()
targets_s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()

# Add 50,000 background distractors
s2_bg = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=25000)
s3_bg = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=25000)

targets_df = pl.concat([targets_s2, targets_s3, s2_bg, s3_bg]).unique(subset=["entity_id"])
print(f"Total target corpus indexed: {len(targets_df):,} records.")

index = BlockingIndex()
for row in targets_df.iter_rows(named=True):
    index.add_target(
        entity_id=row["entity_id"],
        name=row["business_name"],
        addr=row["business_address"],
        country=row["country"]
    )

retrieved_true_links = 0
total_candidates = 0
missed_samples = []

for row in s1_eval.iter_rows(named=True):
    s1_id = row["entity_id"]
    true_targets = true_matches_map[s1_id]
    
    candidates = index.get_candidates(
        s1_id=s1_id,
        name=row["business_name"],
        addr=row["business_address"],
        country=row["country"],
        max_candidates=30,
        min_similarity_floor=25.0
    )
    cand_ids = {c[0] for c in candidates}
    total_candidates += len(cand_ids)
    
    hits = len(true_targets & cand_ids)
    retrieved_true_links += hits
    
    missed = true_targets - cand_ids
    if missed:
        for m_id in missed:
            # find m_id in targets_df
            m_row = targets_df.filter(pl.col("entity_id") == m_id).to_dicts()
            if m_row:
                missed_samples.append((row, m_row[0]))

recall = (retrieved_true_links / total_true_links) * 100
avg_k = total_candidates / len(s1_eval)
print(f"\nExact Recall Ceiling: {recall:.2f}% ({retrieved_true_links:,} / {total_true_links:,})")
print(f"Average Candidates per Entity: {avg_k:.2f}")

print(f"\n--- Analyzing Missed Matches (Total missed: {len(missed_samples)}) ---")
for i, (s1_r, t_r) in enumerate(missed_samples[:8]):
    print(f"\nMissed {i+1}:")
    print(f"  S1 [{s1_r['entity_id']}] ({s1_r['country']}):")
    print(f"    Name: '{s1_r['business_name']}' -> '{normalize_name(s1_r['business_name'])}'")
    print(f"    Addr: '{s1_r['business_address']}' -> '{normalize_address(s1_r['business_address'])}'")
    print(f"  Target [{t_r['entity_id']}] ({t_r['country']}):")
    print(f"    Name: '{t_r['business_name']}' -> '{normalize_name(t_r['business_name'])}'")
    print(f"    Addr: '{t_r['business_address']}' -> '{normalize_address(t_r['business_address'])}'")
