"""
Cross-Validation Module for Business Entity Resolution.
Executes entity-level 5-fold cross-validation, computing mean and standard deviation
for Candidate Recall Ceiling, Macro F_0.5, Precision, Recall, and Singleton Accuracy.
"""

import os
import sys
import time
import numpy as np
import polars as pl
from typing import Dict, List, Set, Tuple

sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from dataset_builder import build_labeled_dataset
from model import EntityResolutionModel
from decision import grid_search_threshold
from evaluation import compute_macro_f05


def run_5fold_cross_validation(
    s1_sample_size: int = 10000,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run 5-fold entity-level cross validation and report performance statistics.
    """
    print(f"=== Starting {n_splits}-Fold Entity-Level Cross Validation ===")
    
    # 1. Load S1 records
    print(f"Loading {s1_sample_size:,} S1 entities...")
    s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=s1_sample_size)
    all_s1_ids = s1_df["entity_id"].to_list()
    
    # 2. Load Ground Truth
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
            
    # 3. Load Targets
    print(f"Fetching {len(all_true_target_ids):,} true targets + background distractors...")
    t_s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(list(all_true_target_ids))
    ).collect()
    t_s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(list(all_true_target_ids))
    ).collect()
    bg_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=30000)
    bg_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=30000)
    targets_df = pl.concat([t_s2, t_s3, bg_s2, bg_s3]).unique(subset=["entity_id"])
    print(f"Target pool: {len(targets_df):,} records.")
    
    # 4. Partition S1 entities into 5 folds
    np.random.seed(random_state)
    shuffled_s1 = np.random.permutation(all_s1_ids)
    fold_splits = np.array_split(shuffled_s1, n_splits)
    
    fold_results = []
    
    for fold in range(n_splits):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        val_entities = set(fold_splits[fold])
        train_entities = set(all_s1_ids) - val_entities
        
        s1_train = s1_df.filter(pl.col("entity_id").is_in(list(train_entities)))
        s1_val = s1_df.filter(pl.col("entity_id").is_in(list(val_entities)))
        
        # Build Train & Val feature sets
        X_train, y_train, tr_s1, tr_target = build_labeled_dataset(
            s1_train, targets_df, gt_map, max_candidates_per_entity=30, min_similarity_floor=24.0, verbose=False
        )
        X_val, y_val, val_s1, val_target = build_labeled_dataset(
            s1_val, targets_df, gt_map, max_candidates_per_entity=30, min_similarity_floor=24.0, verbose=False
        )
        
        # Blocking recall ceiling on validation fold
        val_true_links = sum(len(gt_map.get(sid, set())) for sid in val_entities)
        val_retrieved_links = sum(1 for sid, tid in zip(val_s1, val_target) if tid in gt_map.get(sid, set()))
        recall_ceiling = (val_retrieved_links / val_true_links * 100.0) if val_true_links > 0 else 100.0
        
        # Train model
        model = EntityResolutionModel(num_leaves=41, learning_rate=0.08, n_estimators=150, min_child_samples=25)
        model.fit(X_train, y_train, X_val, y_val)
        
        # Hard negative mining
        weights = model.mine_hard_negatives(X_train, y_train, weight_multiplier=2.5)
        model.fit(X_train, y_train, X_val, y_val, sample_weights=weights)
        
        val_probs = model.predict_proba(X_val)
        
        # Grid search threshold on validation fold with 1-to-1 filter
        best_th, best_f05, metrics = grid_search_threshold(
            pair_s1_ids=val_s1,
            pair_target_ids=val_target,
            probabilities=val_probs,
            ground_truth_map=gt_map,
            all_s1_ids=list(val_entities),
            enforce_one_to_one=True
        )
        
        print(f"Fold {fold+1} Results: Recall Ceiling={recall_ceiling:.2f}%, Optimal tau={best_th:.2f}, "
              f"Macro F_0.5={best_f05:.5f}, Precision={metrics['mean_precision']:.4f}, Recall={metrics['mean_recall']:.4f}")
        
        fold_results.append({
            "fold": fold + 1,
            "recall_ceiling": recall_ceiling,
            "threshold": best_th,
            "macro_f05": best_f05,
            "precision": metrics["mean_precision"],
            "recall": metrics["mean_recall"],
            "singleton_acc": metrics["singleton_accuracy"],
        })
        
    f05_scores = [r["macro_f05"] for r in fold_results]
    recall_ceilings = [r["recall_ceiling"] for r in fold_results]
    precisions = [r["precision"] for r in fold_results]
    recalls = [r["recall"] for r in fold_results]
    
    summary = {
        "mean_f05": float(np.mean(f05_scores)),
        "std_f05": float(np.std(f05_scores)),
        "mean_recall_ceiling": float(np.mean(recall_ceilings)),
        "std_recall_ceiling": float(np.std(recall_ceilings)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "fold_results": fold_results
    }
    
    print("\n================ 5-FOLD CV SUMMARY ================")
    print(f"Mean Macro F_0.5:         {summary['mean_f05']:.5f} +/- {summary['std_f05']:.5f}")
    print(f"Mean Recall Ceiling:      {summary['mean_recall_ceiling']:.2f}% +/- {summary['std_recall_ceiling']:.2f}%")
    print(f"Mean Precision:           {summary['mean_precision']:.4f}")
    print(f"Mean Recall:              {summary['mean_recall']:.4f}")
    print("===================================================")
    return summary

if __name__ == "__main__":
    run_5fold_cross_validation(s1_sample_size=6000, n_splits=5)
