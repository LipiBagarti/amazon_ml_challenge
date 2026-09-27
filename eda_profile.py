import polars as pl
import pandas as pd
import numpy as np
import os
import sys

print("=== Starting EDA Script ===")

# 1. File counts and basic check
train_dir = "dataset/train"
test_dir = "dataset/test"

for split, d in [("Train", train_dir), ("Test", test_dir)]:
    print(f"\n--- {split} Set Overview ---")
    for f in sorted(os.listdir(d)):
        if f.endswith(".tsv"):
            p = os.path.join(d, f)
            sz = os.path.getsize(p) / (1024*1024)
            df = pl.scan_csv(p, separator="\t", ignore_errors=True)
            cnt = df.select(pl.len()).collect().item()
            print(f"{f}: {cnt:,} rows ({sz:.1f} MB)")

# 2. Country distribution and missing values per file
print("\n--- Country Distribution and Null Rates ---")
for split, d in [("Train", train_dir), ("Test", test_dir)]:
    for f in sorted(os.listdir(d)):
        if f.endswith(".tsv") and "ground_truth" not in f:
            p = os.path.join(d, f)
            df = pl.scan_csv(p, separator="\t", ignore_errors=True)
            schema = df.collect_schema()
            
            null_exprs = [pl.col(c).is_null().sum().alias(f"{c}_null") for c in schema.names()]
            nulls = df.select(null_exprs).collect().to_dict(as_series=False)
            total = df.select(pl.len()).collect().item()
            
            country_dist = df.group_by("country").len().collect().sort("len", descending=True)
            print(f"\nFile: {f} (Total: {total:,})")
            for c in schema.names():
                n_null = nulls[f"{c}_null"][0]
                pct = (n_null / total) * 100
                print(f"  Column '{c}': {n_null:,} nulls ({pct:.2f}%)")
            print("  Country counts:")
            for row in country_dist.iter_rows(named=True):
                c_name = row['country']
                c_cnt = row['len']
                print(f"    - {c_name}: {c_cnt:,} ({c_cnt/total*100:.2f}%)")

# 3. Ground Truth Profiling
print("\n--- Ground Truth Profiling ---")
gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
gt_df = pl.read_csv(gt_path, separator="\t")
total_s1 = len(gt_df)
print(f"Total S1 entities in Ground Truth: {total_s1:,}")

# Parse matched_entity_ids
# Handle nulls
gt_df = gt_df.with_columns(
    pl.col("matched_entity_ids").fill_null("")
)

gt_df = gt_df.with_columns(
    pl.when(pl.col("matched_entity_ids") == "")
    .then(pl.lit(0))
    .otherwise(pl.col("matched_entity_ids").str.split(",").list.len())
    .alias("match_count")
)

singletons = gt_df.filter(pl.col("match_count") == 0).height
has_matches = total_s1 - singletons
print(f"Singletons (0 matches): {singletons:,} ({singletons/total_s1*100:.2f}%)")
print(f"Entities with >= 1 match: {has_matches:,} ({has_matches/total_s1*100:.2f}%)")

quantiles = gt_df.select(
    pl.col("match_count").mean().alias("mean"),
    pl.col("match_count").median().alias("median"),
    pl.col("match_count").max().alias("max"),
    pl.col("match_count").quantile(0.75).alias("q75"),
    pl.col("match_count").quantile(0.90).alias("q90"),
    pl.col("match_count").quantile(0.99).alias("q99"),
).collect() if hasattr(gt_df, "collect") else gt_df.select(
    pl.col("match_count").mean().alias("mean"),
    pl.col("match_count").median().alias("median"),
    pl.col("match_count").max().alias("max"),
    pl.col("match_count").quantile(0.75).alias("q75"),
    pl.col("match_count").quantile(0.90).alias("q90"),
    pl.col("match_count").quantile(0.99).alias("q99"),
)
print("Match count statistics per S1 entity:")
for col in quantiles.columns:
    print(f"  {col}: {quantiles[col][0]}")

match_dist = gt_df.group_by("match_count").len().sort("match_count")
print("\nMatch count frequency distribution (first 10):")
for r in match_dist.head(10).iter_rows(named=True):
    mc = r['match_count']
    cnt = r['len']
    print(f"  {mc} matches: {cnt:,} S1 entities ({cnt/total_s1*100:.2f}%)")

# How many unique S2 and S3 matches?
print("\n--- Unique S2 / S3 Matches in GT ---")
# Explode matched_entity_ids
non_empty = gt_df.filter(pl.col("matched_entity_ids") != "").select(
    pl.col("source1_entity_id"),
    pl.col("matched_entity_ids").str.split(",").alias("matched_id")
).explode("matched_id")

total_match_pairs = len(non_empty)
print(f"Total (S1, S2/S3) match links in GT: {total_match_pairs:,}")

unique_matched_s2 = non_empty.filter(pl.col("matched_id").str.starts_with("S2-")).select("matched_id").n_unique()
unique_matched_s3 = non_empty.filter(pl.col("matched_id").str.starts_with("S3-")).select("matched_id").n_unique()
total_unique_matched = non_empty.select("matched_id").n_unique()

print(f"Total unique S2 IDs matched: {unique_matched_s2:,}")
print(f"Total unique S3 IDs matched: {unique_matched_s3:,}")
print(f"Total unique matched targets: {total_unique_matched:,}")

# Check 1-to-1 vs Many-to-1: Does an S2/S3 entity ever match multiple S1 entities?
target_counts = non_empty.group_by("matched_id").len()
multi_claimed_targets = target_counts.filter(pl.col("len") > 1).height
max_claim = target_counts.select(pl.col("len").max()).item()
print(f"S2/S3 IDs claimed by >1 S1 entity: {multi_claimed_targets:,} (Max claims on single target: {max_claim})")

print("\n=== EDA Finished Successfully ===")
