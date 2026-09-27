"""
Cross-Encoder Re-Ranking — score all candidate pairs with a cross-encoder
that jointly attends over both texts, producing much more accurate match
probabilities than independent embeddings.

Includes fine-tuning capability with binary CE / focal loss.
Designed for GPU execution on Google Colab.
"""
import os
import gc
import random
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Optional

try:
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import CrossEncoder, InputExample
except ImportError:
    pass

from normalize import make_record_text


def build_cross_encoder_pairs(
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    candidates: Dict[str, List[Tuple[str, float]]],
    max_negatives_per_positive: int = 5,
    seed: int = 42,
) -> List["InputExample"]:
    """
    Build InputExamples for cross-encoder training.
    label=1 for true matches, label=0 for hard negatives from blocking.
    """
    random.seed(seed)
    s1_lookup = {row["entity_id"]: row for _, row in s1_df.iterrows()}
    t_lookup = {row["entity_id"]: row for _, row in targets_df.iterrows()}

    examples = []
    for s1_id, true_targets in gt_map.items():
        if s1_id not in s1_lookup:
            continue
        s1_row = s1_lookup[s1_id]
        s1_text = make_record_text(
            s1_row.get("business_name", ""),
            s1_row.get("business_address", ""),
            s1_row.get("country", ""),
        )

        # Positives
        for tid in true_targets:
            if tid in t_lookup:
                t_row = t_lookup[tid]
                t_text = make_record_text(
                    t_row.get("business_name", ""),
                    t_row.get("business_address", ""),
                    t_row.get("country", ""),
                )
                examples.append(InputExample(texts=[s1_text, t_text], label=1))

        # Hard negatives from blocking
        cand_ids = [c[0] for c in candidates.get(s1_id, [])]
        false_cands = [c for c in cand_ids if c not in true_targets and c in t_lookup]
        random.shuffle(false_cands)
        n_neg = max_negatives_per_positive * max(len(true_targets), 1)
        for tid in false_cands[:n_neg]:
            t_row = t_lookup[tid]
            t_text = make_record_text(
                t_row.get("business_name", ""),
                t_row.get("business_address", ""),
                t_row.get("country", ""),
            )
            examples.append(InputExample(texts=[s1_text, t_text], label=0))

    random.shuffle(examples)
    print(f"Built {len(examples):,} cross-encoder training examples.")
    return examples


def finetune_cross_encoder(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    examples: List["InputExample"] = None,
    output_dir: str = "models/cross_encoder_finetuned",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
):
    """Fine-tune a cross-encoder on labeled pairs."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading cross-encoder: {model_name}")
    model = CrossEncoder(model_name, num_labels=1, max_length=256)

    warmup_steps = int(len(examples) / batch_size * epochs * warmup_ratio)
    print(f"Fine-tuning for {epochs} epochs, {len(examples):,} examples, "
          f"batch_size={batch_size}, warmup={warmup_steps}")

    model.fit(
        train_dataloader=DataLoader(examples, shuffle=True, batch_size=batch_size),
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=output_dir,
        show_progress_bar=True,
    )

    print(f"Fine-tuned cross-encoder saved to {output_dir}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


def rerank_candidates(
    model_or_path: str,
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    candidates: Dict[str, List[Tuple[str, float]]],
    batch_size: int = 256,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Re-rank all candidates using the cross-encoder.
    Returns {s1_id: [(target_id, cross_encoder_score), ...]} sorted desc.
    """
    print(f"Loading cross-encoder from: {model_or_path}")
    model = CrossEncoder(model_or_path, max_length=256)

    s1_lookup = {row["entity_id"]: row for _, row in s1_df.iterrows()}
    t_lookup = {row["entity_id"]: row for _, row in targets_df.iterrows()}

    # Build all pairs for scoring
    all_pairs = []         # (s1_text, t_text)
    all_pair_keys = []     # (s1_id, t_id)

    for s1_id, cand_list in candidates.items():
        if s1_id not in s1_lookup:
            continue
        s1_row = s1_lookup[s1_id]
        s1_text = make_record_text(
            s1_row.get("business_name", ""),
            s1_row.get("business_address", ""),
            s1_row.get("country", ""),
        )
        for tid, _ in cand_list:
            if tid not in t_lookup:
                continue
            t_row = t_lookup[tid]
            t_text = make_record_text(
                t_row.get("business_name", ""),
                t_row.get("business_address", ""),
                t_row.get("country", ""),
            )
            all_pairs.append([s1_text, t_text])
            all_pair_keys.append((s1_id, tid))

    print(f"Scoring {len(all_pairs):,} candidate pairs with cross-encoder...")
    if not all_pairs:
        return {}

    # Score in batches
    scores = model.predict(all_pairs, batch_size=batch_size, show_progress_bar=True)
    # Sigmoid if raw logits
    scores = 1.0 / (1.0 + np.exp(-np.array(scores)))

    # Group by S1
    reranked: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, tid), score in zip(all_pair_keys, scores):
        if s1_id not in reranked:
            reranked[s1_id] = []
        reranked[s1_id].append((tid, float(score)))

    # Sort each entity's candidates descending
    for s1_id in reranked:
        reranked[s1_id].sort(key=lambda x: x[1], reverse=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reranked
