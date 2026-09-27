import sys
import os
import time
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from blocking import BlockingIndex

print("=== Evaluating Candidate Generation Recall Ceiling ===")

# 1. Load Ground Truth sample
print("Loading Ground Truth...")
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=100000)
# Filter entities that have matches
gt_matches = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")).head(5000)

sample_s1_ids = set(gt_matches["source1_entity_id"].to_list())

# Ground truth map: s1_id -> set of true matched target IDs
true_matches_map = {}
all_true_target_ids = set()
total_true_links = 0

for row in gt_matches.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    targets = set(row["matched_entity_ids"].split(","))
    true_matches_map[s1_id] = targets
    all_true_target_ids.update(targets)
    total_true_links += len(targets)

print(f"Sampled {len(sample_s1_ids):,} S1 entities with {total_true_links:,} true target links.")

# 2. Load S1 data for these entities
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=150000)
s1_eval = s1_df.filter(pl.col("entity_id").is_in(list(sample_s1_ids)))
print(f"Loaded {len(s1_eval):,} S1 query records.")

# 3. Load Targets: Include ALL true targets + 100,000 background distractors
print("Loading target records (S2 + S3) with background distractors...")
s2_df = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=150000)
s3_df = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=150000)
targets_df = pl.concat([s2_df, s3_df])

# Ensure all true targets from our sample are in targets_df
# If some true targets were beyond the first 150k rows, fetch them specifically
missing_targets = all_true_target_ids - set(targets_df["entity_id"].to_list())
if missing_targets:
    print(f"Fetching {len(missing_targets)} true targets beyond initial chunk...")
    # Read needed targets
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        chunk_df = pl.scan_csv(f"dataset/train/{s_file}", separator="\t").filter(
            pl.col("entity_id").is_in(list(missing_targets))
        ).collect()
        if len(chunk_df) > 0:
            targets_df = pl.concat([targets_df, chunk_df])
            missing_targets -= set(chunk_df["entity_id"].to_list())

print(f"Total target corpus indexed: {len(targets_df):,} records.")

# 4. Build Index
print("\nBuilding Blocking Inverted Index...")
t0 = time.time()
index = BlockingIndex()

for row in targets_df.iter_rows(named=True):
    index.add_target(
        entity_id=row["entity_id"],
        name=row["business_name"],
        addr=row["business_address"],
        country=row["country"]
    )
build_time = time.time() - t0
print(f"Index built in {build_time:.2f}s ({len(targets_df)/build_time:,.0f} records/sec).")

# 5. Query Candidates and Measure Recall Ceiling
print("\nQuerying candidates for evaluation S1 entities...")
t0 = time.time()

retrieved_true_links = 0
total_candidates_generated = 0
entity_recalls = []

s1_eval_rows = s1_eval.to_dicts()
for row in s1_eval_rows:
    s1_id = row["entity_id"]
    true_targets = true_matches_map.get(s1_id, set())
    
    candidates = index.get_candidates(
        s1_id=s1_id,
        name=row["business_name"],
        addr=row["business_address"],
        country=row["country"],
        max_candidates=30,
        min_similarity_floor=30.0
    )
    
    cand_ids = {c[0] for c in candidates}
    total_candidates_generated += len(cand_ids)
    
    hits = len(true_targets & cand_ids)
    retrieved_true_links += hits
    
    if len(true_targets) > 0:
        entity_recalls.append(hits / len(true_targets))

query_time = time.time() - t0
query_rate = len(s1_eval_rows) / query_time

overall_recall = (retrieved_true_links / total_true_links) * 100
mean_entity_recall = (sum(entity_recalls) / len(entity_recalls)) * 100
avg_candidates_per_entity = total_candidates_generated / len(s1_eval_rows)

print("\n================ BLOCKING RESULTS ================")
print(f"Total True Links in Evaluation Set:  {total_true_links:,}")
print(f"Retrieved True Links in Candidates:  {retrieved_true_links:,}")
print(f"Missed True Links:                   {total_true_links - retrieved_true_links:,}")
print(f"Candidate-Set Recall Ceiling:        {overall_recall:.2f}%")
print(f"Mean Entity-Level Recall Ceiling:    {mean_entity_recall:.2f}%")
print(f"Average Candidates per Entity (K):   {avg_candidates_per_entity:.2f}")
print(f"Query Throughput:                    {query_rate:,.0f} queries/sec")
print(f"Target Space Reduction Ratio:        {100 * (1 - avg_candidates_per_entity / len(targets_df)):.6f}%")
print("==================================================")
