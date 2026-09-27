"""
Scoring Module — exact macro-averaged F_0.5 as specified by the competition.
"""
from typing import Dict, Set, List, Optional
import numpy as np


def f05_single(pred: Set[str], true: Set[str]) -> float:
    """F_0.5 for one Source-1 entity."""
    if not true:                       # singleton
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(
    preds: Dict[str, Set[str]],
    truths: Dict[str, Set[str]],
    all_ids: Optional[List[str]] = None,
) -> dict:
    """
    Macro-averaged F_0.5 across all Source-1 entities.
    Returns dict with macro_f05, mean_precision, mean_recall, singleton_accuracy.
    """
    if all_ids is None:
        all_ids = sorted(set(truths.keys()) | set(preds.keys()))

    scores, precs, recs = [], [], []
    sing_total = sing_ok = 0

    for sid in all_ids:
        t = truths.get(sid, set())
        p = preds.get(sid, set())
        scores.append(f05_single(p, t))

        if not t:
            sing_total += 1
            if not p:
                sing_ok += 1
        else:
            tp = len(p & t)
            precs.append(tp / len(p) if p else 0.0)
            recs.append(tp / len(t))

    return {
        "macro_f05": float(np.mean(scores)),
        "mean_precision": float(np.mean(precs)) if precs else 0.0,
        "mean_recall": float(np.mean(recs)) if recs else 0.0,
        "singleton_accuracy": (sing_ok / sing_total) if sing_total else 1.0,
        "n_entities": len(all_ids),
        "n_singletons": sing_total,
    }
