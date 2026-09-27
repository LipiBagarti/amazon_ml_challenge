"""
Model Module for Business Entity Resolution.
Trains LightGBM classifier with early stopping, performs hard-negative mining,
and outputs calibrated probabilities for candidate pairs.
"""

import os
import sys
import numpy as np
import lightgbm as lgb
from typing import Dict, Tuple, Optional, List

sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from features import FEATURE_NAMES


class EntityResolutionModel:
    """
    Gradient Boosted Matching Model using LightGBM.
    Tuned for precision-biased binary classification on tabular string similarities.
    """
    def __init__(
        self,
        num_leaves: int = 41,
        learning_rate: float = 0.08,
        n_estimators: int = 150,
        min_child_samples: int = 25,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        scale_pos_weight: float = 1.0,
        random_state: int = 42
    ):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.feature_importances_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None
    ):
        """Fit LightGBM model on training feature matrix."""
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        
        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)] if eval_set else None
        )
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive class probability P(match = 1)."""
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def get_feature_importances(self) -> List[Tuple[str, float]]:
        """Return sorted list of (feature_name, importance)."""
        if self.feature_importances_ is None:
            return []
        feats = list(zip(FEATURE_NAMES, self.feature_importances_))
        feats.sort(key=lambda x: x[1], reverse=True)
        return feats

    def mine_hard_negatives(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weight_multiplier: float = 2.5
    ) -> np.ndarray:
        """
        Identify false positives (negatives with high predicted probability P > 0.45)
        and construct an upweighted sample_weight array to sharpen the precision boundary.
        """
        preds = self.predict_proba(X_train)
        weights = np.ones(len(y_train), dtype=np.float32)
        
        # Hard negatives: ground truth is 0, but model predicted > 0.45
        hard_neg_mask = (y_train == 0) & (preds > 0.45)
        weights[hard_neg_mask] = weight_multiplier
        num_hard_negs = int(hard_neg_mask.sum())
        print(f"Mined {num_hard_negs:,} hard negative pairs (upweighted {weight_multiplier}x).")
        return weights
