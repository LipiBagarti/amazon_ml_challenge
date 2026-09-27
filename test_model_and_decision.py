import sys
import os
import time
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from dataset_builder import build_labeled_dataset
from model import EntityResolutionModel
from decision import apply_decision_rules, grid_search_threshold
from evaluation import compute_macro_f05
from features import FEATURE_NAMES

print("=== Running Model Training & Precision Decision Optimization ===")

# 1. Load Data & Prepare Splits
print("Loading data for train/validation...")
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=8000)
all_s1_ids = s1_df["entity_id"].to_list()

np.random.seed(42)
shuffled_ids = np.random.permutation(all_s1_ids)
split_idx = int(len(shuffled_ids) * 0.75)
train_ids = set(shuffled_ids[:split_idx])
val_ids = set(shuffled_ids[split_idx:])

s1_train = s1_df.filter(pl.col("entity_id").is_in(list(train_ids)))
s1_val = s1_df.filter(pl.col("entity_id").is_in(list(val_ids)))

# Ground truth map
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

# Target pool
t_s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()
t_s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect()
bg_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=25000)
bg_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=25000)
targets_df = pl.concat([t_s2, t_s3, bg_s2, bg_s3]).unique(subset=["entity_id"])

print(f"Entities: Train={len(train_ids):,}, Val={len(val_ids):,}. Target pool={len(targets_df):,}")

# Build Feature Matrices
print("\nConstructing Training Feature Matrix...")
X_train, y_train, train_s1_pairs, train_target_pairs = build_labeled_dataset(
    s1_train, targets_df, gt_map, max_candidates_per_entity=30, min_similarity_floor=24.0, verbose=False
)
print(f"X_train: {X_train.shape} (Positives: {int(y_train.sum()):,}, Negatives: {int((1-y_train).sum()):,})")

print("Constructing Validation Feature Matrix...")
X_val, y_val, val_s1_pairs, val_target_pairs = build_labeled_dataset(
    s1_val, targets_df, gt_map, max_candidates_per_entity=30, min_similarity_floor=24.0, verbose=False
)
print(f"X_val:   {X_val.shape} (Positives: {int(y_val.sum()):,}, Negatives: {int((1-y_val).sum()):,})")

# 2. Train LightGBM Baseline
print("\n--- Training LightGBM Baseline ---")
t0 = time.time()
er_model = EntityResolutionModel(
    num_leaves=41,
    learning_rate=0.08,
    n_estimators=150,
    min_child_samples=25,
    scale_pos_weight=1.2,
    random_state=42
)
er_model.fit(X_train, y_train, X_val, y_val)
train_time = time.time() - t0
print(f"LightGBM trained in {train_time:.2f}s.")

# Raw pairwise probabilities
val_probs_base = er_model.predict_proba(X_val)
auc_base = roc_auc_score(y_val, val_probs_base)
pr_auc_base = average_precision_score(y_val, val_probs_base)
print(f"Pairwise ROC-AUC: {auc_base:.4f}, PR-AUC: {pr_auc_base:.4f}")

# Top 10 Feature Importances
print("\nTop 10 Feature Importances:")
for fname, imp in er_model.get_feature_importances()[:10]:
    print(f"  {fname:<25}: {imp:.1f}")

# 3. Round 1 of Hard Negative Mining
print("\n--- Hard Negative Mining Round ---")
sample_weights = er_model.mine_hard_negatives(X_train, y_train, weight_multiplier=2.5)

# Retrain with hard negative weights
er_model_refined = EntityResolutionModel(
    num_leaves=41,
    learning_rate=0.07,
    n_estimators=180,
    min_child_samples=25,
    scale_pos_weight=1.0,
    random_state=42
)
er_model_refined.fit(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
val_probs = er_model_refined.predict_proba(X_val)

auc_refined = roc_auc_score(y_val, val_probs)
pr_auc_refined = average_precision_score(y_val, val_probs)
print(f"Refined Model Pairwise ROC-AUC: {auc_refined:.4f}, PR-AUC: {pr_auc_refined:.4f}")

# 4. Compare Decision Layers & Threshold Optimization
val_s1_list = list(val_ids)

print("\n--- Comparing Decision Rules against Macro F_0.5 ---")

# Experiment A: Without 1-to-1 filter (Independent Pairwise Cutoff)
best_th_no121, best_f05_no121, metrics_no121 = grid_search_threshold(
    pair_s1_ids=val_s1_pairs,
    pair_target_ids=val_target_pairs,
    probabilities=val_probs,
    ground_truth_map=gt_map,
    all_s1_ids=val_s1_list,
    enforce_one_to_one=False
)
print(f"\n[A] Independent Pairwise (No 1-to-1 Filter):")
print(f"  Optimal Threshold:   tau = {best_th_no121:.2f}")
print(f"  Macro F_0.5 Score:   {best_f05_no121:.5f}")
print(f"  Mean Precision:      {metrics_no121['mean_precision']:.4f}")
print(f"  Mean Recall:         {metrics_no121['mean_recall']:.4f}")
print(f"  Singleton Accuracy:  {metrics_no121['singleton_accuracy']:.4f}")

# Experiment B: WITH Global 1-to-1 Bipartite Greedy Filter
best_th_121, best_f05_121, metrics_121 = grid_search_threshold(
    pair_s1_ids=val_s1_pairs,
    pair_target_ids=val_target_pairs,
    probabilities=val_probs,
    ground_truth_map=gt_map,
    all_s1_ids=val_s1_list,
    enforce_one_to_one=True
)
print(f"\n[B] Global 1-to-1 Bipartite Greedy Filter:")
print(f"  Optimal Threshold:   tau = {best_th_121:.2f}")
print(f"  Macro F_0.5 Score:   {best_f05_121:.5f}")
print(f"  Mean Precision:      {metrics_121['mean_precision']:.4f}")
print(f"  Mean Recall:         {metrics_121['mean_recall']:.4f}")
print(f"  Singleton Accuracy:  {metrics_121['singleton_accuracy']:.4f}")

gain = (best_f05_121 - best_f05_no121) * 100
print(f"\n>>> Net Score Gain from Global 1-to-1 Resolution: +{gain:.3f}% F_0.5! <<<")
