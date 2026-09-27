"""
Optimal one-to-one match resolution via the Hungarian (Munkres) algorithm.

Since each S2/S3 record corresponds to at most one real business,
when multiple S1 entities claim the same candidate above threshold,
this module finds the globally optimal assignment that maximises
total confidence while enforcing 1-to-1 constraints.
"""
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_resolve(
    pair_s1_ids: List[str],
    pair_target_ids: List[str],
    probabilities: np.ndarray,
    threshold: float = 0.65,
    max_matches_per_entity: int = 11,
) -> Dict[str, Set[str]]:
    """
    1. Keep only pairs with P(match) >= threshold.
    2. Identify conflicting claims (multiple S1 entities want the same target).
    3. Resolve conflicts via Hungarian algorithm on a per-target basis.
    4. Cap matches per S1 entity.
    """
    # Step 1: gather above-threshold pairs, grouped by target
    target_claims: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for s1, tid, p in zip(pair_s1_ids, pair_target_ids, probabilities):
        if p >= threshold:
            target_claims[tid].append((s1, float(p)))

    # Step 2: resolve — for non-conflicting targets, just accept;
    #         for conflicting targets, give to highest-confidence claimant
    accepted: Dict[str, Set[str]] = defaultdict(set)

    # Collect all conflicting targets per group of S1 claimants
    # Simple greedy-by-confidence (globally optimal for disjoint conflicts):
    # Sort ALL (s1, target, prob) triples by prob descending,
    # greedily accept each if the target hasn't been claimed yet.
    all_triples = []
    for tid, claims in target_claims.items():
        for s1, p in claims:
            all_triples.append((p, s1, tid))
    all_triples.sort(reverse=True)  # highest confidence first

    claimed_targets: Set[str] = set()
    entity_match_count: Dict[str, int] = defaultdict(int)

    for prob, s1, tid in all_triples:
        if tid in claimed_targets:
            continue
        if entity_match_count[s1] >= max_matches_per_entity:
            continue
        accepted[s1].add(tid)
        claimed_targets.add(tid)
        entity_match_count[s1] += 1

    return dict(accepted)


def hungarian_resolve_optimal(
    pair_s1_ids: List[str],
    pair_target_ids: List[str],
    probabilities: np.ndarray,
    threshold: float = 0.65,
    max_matches_per_entity: int = 11,
) -> Dict[str, Set[str]]:
    """
    Full Hungarian (optimal) resolution for conflicting claims.
    Falls back to greedy for non-conflicting targets.
    More expensive but finds the globally optimal assignment.
    """
    # Gather above-threshold pairs
    target_claims: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for s1, tid, p in zip(pair_s1_ids, pair_target_ids, probabilities):
        if p >= threshold:
            target_claims[tid].append((s1, float(p)))

    # Separate non-conflicting (single claimant) and conflicting targets
    accepted: Dict[str, Set[str]] = defaultdict(set)
    conflicting_targets: Dict[str, List[Tuple[str, float]]] = {}

    for tid, claims in target_claims.items():
        if len(claims) == 1:
            s1, prob = claims[0]
            accepted[s1].add(tid)
        else:
            conflicting_targets[tid] = claims

    # Resolve conflicts with Hungarian algorithm
    if conflicting_targets:
        # Build cost matrix: rows=unique S1 claimants, cols=conflicting targets
        all_s1 = sorted({s1 for claims in conflicting_targets.values() for s1, _ in claims})
        all_tids = sorted(conflicting_targets.keys())
        s1_idx = {s: i for i, s in enumerate(all_s1)}
        tid_idx = {t: j for j, t in enumerate(all_tids)}

        n_rows, n_cols = len(all_s1), len(all_tids)
        # Cost = negative probability (we minimise cost → maximise prob)
        cost = np.full((n_rows, n_cols), 1e6, dtype=np.float64)

        for tid, claims in conflicting_targets.items():
            j = tid_idx[tid]
            for s1, prob in claims:
                i = s1_idx[s1]
                cost[i, j] = -prob  # negative so minimising = maximising

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0:  # only accept real claims, not dummy entries
                s1 = all_s1[r]
                tid = all_tids[c]
                accepted[s1].add(tid)

    # Apply max_matches_per_entity cap
    for s1 in list(accepted.keys()):
        if len(accepted[s1]) > max_matches_per_entity:
            # Keep highest-confidence ones
            s1_pairs = []
            for tid in accepted[s1]:
                # Find the probability for this pair
                for s1_id, t_id, p in zip(pair_s1_ids, pair_target_ids, probabilities):
                    if s1_id == s1 and t_id == tid:
                        s1_pairs.append((tid, p))
                        break
            s1_pairs.sort(key=lambda x: x[1], reverse=True)
            accepted[s1] = {tid for tid, _ in s1_pairs[:max_matches_per_entity]}

    return dict(accepted)
