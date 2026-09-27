"""
Evaluation Module for Business Entity Resolution.
Implements the exact macro-averaged F_0.5 metric specified by the competition.
"""

from typing import Dict, Set, List, Optional
import numpy as np


def compute_entity_f05(predicted_set: Set[str], true_set: Set[str]) -> float:
    """
    Compute F_0.5 score for a single Source-1 entity.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    - Singletons (true_set is empty):
        - Score = 1.0 if predicted_set is empty
        - Score = 0.0 if predicted_set is non-empty
    - Non-singletons:
        - Score = 0.0 if predicted_set is empty or TP == 0
    """
    # Singleton case
    if len(true_set) == 0:
        return 1.0 if len(predicted_set) == 0 else 0.0
        
    # Non-singleton but predicted empty
    if len(predicted_set) == 0:
        return 0.0
        
    tp = len(predicted_set & true_set)
    if tp == 0:
        return 0.0
        
    precision = tp / len(predicted_set)
    recall = tp / len(true_set)
    
    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0
        
    f05 = (1.25 * precision * recall) / denominator
    return f05


def compute_macro_f05(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    all_s1_ids: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5 across all Source-1 entities.
    Every entity in all_s1_ids (or ground_truth_map) must be evaluated.
    """
    if all_s1_ids is None:
        all_s1_ids = list(ground_truth_map.keys())
        
    scores = []
    precisions = []
    recalls = []
    
    singleton_count = 0
    singleton_correct = 0
    
    for s1_id in all_s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predictions_map.get(s1_id, set())
        
        score = compute_entity_f05(pred_set, true_set)
        scores.append(score)
        
        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            if len(pred_set) > 0:
                tp = len(pred_set & true_set)
                precisions.append(tp / len(pred_set))
                recalls.append(tp / len(true_set))
            else:
                precisions.append(0.0)
                recalls.append(0.0)
                
    macro_f05 = float(np.mean(scores))
    mean_prec = float(np.mean(precisions)) if precisions else 0.0
    mean_rec = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = (singleton_correct / singleton_count) if singleton_count > 0 else 1.0
    
    return {
        "macro_f05": macro_f05,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "singleton_accuracy": singleton_acc,
        "total_entities": len(all_s1_ids),
        "total_singletons": singleton_count,
    }
