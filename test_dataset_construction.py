import sys
import os
import time
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from dataset_builder import build_labeled_dataset
from features import FEATURE_NAMES

print("=== Testing Training & Validation Dataset Construction ===")

# 1. Load S1 sample (e.g. 6,000 entities)
print("Loading Source 1 records...")
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=6000)
all_s1_ids = s1_df["entity_id"].to_list()

# 2. Strict Entity-Level 80/20 Train/Validation Split
np.random.seed(42)
shuffled_ids = np.random.permutation(all_s1_ids)
split_idx = int(len(shuffled_ids) * 0.8)
train_ids = set(shuffled_ids[:split_idx])
val_ids = set(shuffled_ids[split_idx:])

assert len(train_ids & val_ids) == 0, "ERROR: Entity leakage detected!"
print(f"Entity split: Train = {len(train_ids):,} entities (80%), Val = {len(val_ids):,} entities (20%). Entity overlap = 0.")

s1_train = s1_df.filter(pl.col("entity_id").is_in(list(train_ids)))
s1_val = s1_df.filter(pl.col("entity_id").is_in(list(val_ids)))

# 3. Load Ground Truth map for these entities
print("Loading Ground Truth...")
gt_df = pl.scan_csv("dataset/train/train_ground_truth.tsv", separator="\t").filter(
    pl.col("source1_entity_id").is_in(all_s1_ids)
).collect()

gt_map = {}
all_true_target_ids = set()
for r in gt_df.iter_rows(named=True):
    s1_id = r["source1_entity_id"]
    targets_str = r["matched_entity_ids"]
    if targets_str and str(targets_str).strip() and str(targets_str).lower() != "nan":
        targets = set(targets_str.split(","))
        gt_map[s1_id] = targets
        all_true_target_ids.update(targets)
    else:
        gt_map[s1_id] = set()

# 4. Fetch target corpus (all true targets + background distractors)
print(f"Fetching {len(all_true_target_ids):,} true targets...")
t_s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()
t_s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()

# Background distractors
bg_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=20000)
bg_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=20000)

targets_df = pl.concat([t_s2, t_s3, bg_s2, bg_s3]).unique(subset=["entity_id"])
print(f"Total target corpus: {len(targets_df):,} records.")

# 5. Build Train Dataset
print("\n--- Constructing Train Set ---")
X_train, y_train, train_s1, train_targets = build_labeled_dataset(
    s1_df=s1_train,
    targets_df=targets_df,
    gt_map=gt_map,
    max_candidates_per_entity=30,
    min_similarity_floor=25.0
)

# 6. Build Validation Dataset
print("\n--- Constructing Validation Set ---")
X_val, y_val, val_s1, val_targets = build_labeled_dataset(
    s1_df=s1_val,
    targets_df=targets_df,
    gt_map=gt_map,
    max_candidates_per_entity=30,
    min_similarity_floor=25.0
)

print("\n================ DATASET CONSTRUCTION SUMMARY ================")
print(f"Feature Names ({len(FEATURE_NAMES)} total): {FEATURE_NAMES}")
print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_val Shape:   {X_val.shape}, y_val Shape:   {y_val.shape}")
print(f"Train Positives: {int(y_train.sum()):,} ({y_train.mean()*100:.2f}%), Negatives: {int((1-y_train).sum()):,}")
print(f"Val Positives:   {int(y_val.sum()):,} ({y_val.mean()*100:.2f}%), Negatives: {int((1-y_val).sum()):,}")
print("No NaN values in X_train:", not np.isnan(X_train).any())
print("No NaN values in X_val:  ", not np.isnan(X_val).any())
print("==============================================================")
