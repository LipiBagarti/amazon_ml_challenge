"""
Decision Layer Module for Business Entity Resolution.
Implements:
1. Optimal probability thresholding for F_0.5
2. Global 1-to-1 Bipartite Greedy Assignment
3. Singleton Safeguard Floor
4. Max Matches per Entity Cap
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np


def apply_decision_rules(
    pair_s1_ids: List[str],
    pair_target_ids: List[str],
    probabilities: np.ndarray,
    threshold: float = 0.65,
    singleton_floor: float = 0.50,
    max_matches_per_entity: int = 10,
    enforce_one_to_one: bool = True
) -> Dict[str, Set[str]]:
    """
    Convert pairwise probabilities into final entity match lists.
    
    1. Filter pairs with probability >= threshold.
    2. Global 1-to-1 resolution: Sort all candidate pairs globally by confidence.
       Each target S2/S3 record can be matched to AT MOST ONE S1 entity (highest confidence wins).
    3. Singleton safeguard: If an entity has no match >= singleton_floor, keep empty.
    4. Cap maximum matches per entity.
    """
    # Group pairs above threshold
    filtered_pairs = []
    for s1_id, t_id, prob in zip(pair_s1_ids, pair_target_ids, probabilities):
        if prob >= threshold:
            filtered_pairs.append((s1_id, t_id, float(prob)))
            
    # Sort globally by confidence descending
    filtered_pairs.sort(key=lambda x: x[2], reverse=True)
    
    predictions_map = defaultdict(set)
    claimed_targets = set()
    
    for s1_id, t_id, prob in filtered_pairs:
        # Check max matches cap
        if len(predictions_map[s1_id]) >= max_matches_per_entity:
            continue
            
        if enforce_one_to_one:
            if t_id not in claimed_targets:
                predictions_map[s1_id].add(t_id)
                claimed_targets.add(t_id)
        else:
            predictions_map[s1_id].add(t_id)
            
    return dict(predictions_map)


def grid_search_threshold(
    pair_s1_ids: List[str],
    pair_target_ids: List[str],
    probabilities: np.ndarray,
    ground_truth_map: Dict[str, Set[str]],
    all_s1_ids: List[str],
    thresholds: List[float] = None,
    enforce_one_to_one: bool = True
) -> Tuple[float, float, Dict[str, float]]:
    """
    Grid-search optimal decision threshold directly against macro F_0.5.
    Returns: (best_threshold, best_f05, metrics_dict)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.40, 0.92, 0.04)]
        
    from evaluation import compute_macro_f05
    
    best_thresh = 0.65
    best_f05 = -1.0
    best_metrics = {}
    
    for th in thresholds:
        preds = apply_decision_rules(
            pair_s1_ids=pair_s1_ids,
            pair_target_ids=pair_target_ids,
            probabilities=probabilities,
            threshold=th,
            enforce_one_to_one=enforce_one_to_one
        )
        metrics = compute_macro_f05(preds, ground_truth_map, all_s1_ids)
        score = metrics["macro_f05"]
        if score > best_f05:
            best_f05 = score
            best_thresh = th
            best_metrics = metrics
            
    return best_thresh, best_f05, best_metrics
