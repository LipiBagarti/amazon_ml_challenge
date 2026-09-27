"""
GBM Training with Optuna Hyperparameter Optimization.

Stacks the cross-encoder score with cheap lexical/phonetic features into
a LightGBM classifier, tuned directly against macro-F_0.5.

Designed for CPU (LightGBM doesn't need GPU).
"""
import os
import gc
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict

import lightgbm as lgb
import optuna

from features import compute_features, FEATURE_NAMES
from scoring import macro_f05
from assignment_hungarian import hungarian_resolve


def build_training_data(
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    candidates: Dict[str, List[Tuple[str, float]]],
    cross_encoder_scores: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build feature matrix + labels from candidate pairs.
    Features = lexical/phonetic features + cross_encoder_score.
    """
    s1_lookup = {row["entity_id"]: row for _, row in s1_df.iterrows()}
    t_lookup = {row["entity_id"]: row for _, row in targets_df.iterrows()}

    # Build cross-encoder score lookup
    ce_lookup = {}
    if cross_encoder_scores:
        for s1_id, scores in cross_encoder_scores.items():
            for tid, score in scores:
                ce_lookup[(s1_id, tid)] = score

    features_list = []
    labels = []
    pair_s1_ids = []
    pair_target_ids = []

    for s1_id, cand_list in candidates.items():
        if s1_id not in s1_lookup:
            continue
        s1_row = s1_lookup[s1_id]
        true_set = gt_map.get(s1_id, set())
        n_cands = len(cand_list)

        for rank, (tid, blocking_score) in enumerate(cand_list):
            if tid not in t_lookup:
                continue
            t_row = t_lookup[tid]

            feats = compute_features(
                s1_name=s1_row.get("business_name", ""),
                s1_addr=s1_row.get("business_address", ""),
                s1_country=s1_row.get("country", ""),
                t_name=t_row.get("business_name", ""),
                t_addr=t_row.get("business_address", ""),
                t_country=t_row.get("country", ""),
                target_id=tid,
                blocking_score=blocking_score,
                candidate_rank=rank,
                total_candidates=n_cands,
            )
            # Append cross-encoder score as an extra feature
            ce_score = ce_lookup.get((s1_id, tid), 0.0)
            feats.append(ce_score)

            features_list.append(feats)
            labels.append(1 if tid in true_set else 0)
            pair_s1_ids.append(s1_id)
            pair_target_ids.append(tid)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Built training data: {X.shape[0]:,} pairs, "
          f"{y.sum():,} positives ({100*y.mean():.2f}%), "
          f"{len(FEATURE_NAMES)+1} features (incl. cross-encoder)")
    return X, y, pair_s1_ids, pair_target_ids


def train_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    s1_ids_train: List[str], target_ids_train: List[str],
    X_val: np.ndarray, y_val: np.ndarray,
    s1_ids_val: List[str], target_ids_val: List[str],
    gt_map: Dict[str, Set[str]],
    params: dict,
    threshold: float = 0.5,
    max_matches: int = 11,
) -> Tuple[lgb.Booster, float, dict]:
    """Train LightGBM and evaluate on validation set using macro F_0.5."""

    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / max(pos_count, 1)

    full_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_threads": -1,
        "scale_pos_weight": spw,
        **params,
    }

    feature_names = FEATURE_NAMES + ["cross_encoder_score"]
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)

    model = lgb.train(
        full_params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Predict and evaluate
    probs = model.predict(X_val)
    preds_dict = hungarian_resolve(
        s1_ids_val, target_ids_val, probs,
        threshold=threshold, max_matches_per_entity=max_matches,
    )
    result = macro_f05(preds_dict, gt_map,
                       all_ids=sorted(set(s1_ids_val)))

    return model, result["macro_f05"], result


def optuna_objective(
    trial: "optuna.Trial",
    X_train: np.ndarray, y_train: np.ndarray,
    s1_ids_train: List[str], target_ids_train: List[str],
    X_val: np.ndarray, y_val: np.ndarray,
    s1_ids_val: List[str], target_ids_val: List[str],
    gt_map: Dict[str, Set[str]],
) -> float:
    """Optuna objective: tune GBM hyperparams + decision threshold."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    threshold = trial.suggest_float("threshold", 0.3, 0.9)
    max_matches = trial.suggest_int("max_matches", 3, 15)

    _, score, _ = train_evaluate(
        X_train, y_train, s1_ids_train, target_ids_train,
        X_val, y_val, s1_ids_val, target_ids_val,
        gt_map, params, threshold, max_matches,
    )
    return score


def run_optuna_tuning(
    X_train: np.ndarray, y_train: np.ndarray,
    s1_ids_train: List[str], target_ids_train: List[str],
    X_val: np.ndarray, y_val: np.ndarray,
    s1_ids_val: List[str], target_ids_val: List[str],
    gt_map: Dict[str, Set[str]],
    n_trials: int = 100,
    study_name: str = "entity_resolution",
    save_dir: str = "models",
) -> Tuple[dict, float]:
    """Run Optuna study and return best params + score."""
    os.makedirs(save_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: optuna_objective(
            trial,
            X_train, y_train, s1_ids_train, target_ids_train,
            X_val, y_val, s1_ids_val, target_ids_val,
            gt_map,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial: #{best.number}, macro F_0.5 = {best.value:.5f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")
    print(f"{'='*60}")

    # Save study
    with open(os.path.join(save_dir, "optuna_best_params.json"), "w") as f:
        json.dump({"best_params": best.params, "best_score": best.value}, f, indent=2)

    return best.params, best.value


def train_final_model(
    X: np.ndarray, y: np.ndarray,
    best_params: dict,
    save_path: str = "models/lgbm_final.txt",
) -> lgb.Booster:
    """Train final model on ALL data with best hyperparams."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    threshold = best_params.pop("threshold", 0.5)
    max_matches = best_params.pop("max_matches", 11)

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    spw = neg_count / max(pos_count, 1)

    full_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_threads": -1,
        "scale_pos_weight": spw,
        **best_params,
    }

    feature_names = FEATURE_NAMES + ["cross_encoder_score"]
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_names)
    model = lgb.train(full_params, dtrain, num_boost_round=1000)
    model.save_model(save_path)

    config = {"threshold": threshold, "max_matches": int(max_matches)}
    with open(save_path.replace(".txt", "_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Final model saved to {save_path}")
    print(f"Decision config: threshold={threshold:.4f}, max_matches={int(max_matches)}")
    return model
