"""
Bi-Encoder Fine-Tuning on labelled entity-resolution pairs.
Uses MultipleNegativesRankingLoss so that true-match records are pulled
together in embedding space while hard negatives are pushed apart.

Designed for GPU execution on Google Colab.
"""
import os
import gc
import random
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple

try:
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import (
        SentenceTransformer, InputExample, losses, evaluation
    )
except ImportError:
    pass

from normalize import make_record_text


def build_training_pairs(
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    candidates: Dict[str, List[Tuple[str, float]]],
    max_negatives_per_positive: int = 3,
    seed: int = 42,
) -> List["InputExample"]:
    """
    Build (anchor, positive) and (anchor, negative) InputExamples
    from ground truth and blocking candidates.
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

        # Positive pairs
        for tid in true_targets:
            if tid not in t_lookup:
                continue
            t_row = t_lookup[tid]
            t_text = make_record_text(
                t_row.get("business_name", ""),
                t_row.get("business_address", ""),
                t_row.get("country", ""),
            )
            examples.append(InputExample(texts=[s1_text, t_text], label=1.0))

        # Hard negatives from blocking candidates that are NOT true matches
        cand_ids = [c[0] for c in candidates.get(s1_id, [])]
        false_cands = [c for c in cand_ids if c not in true_targets and c in t_lookup]
        random.shuffle(false_cands)
        for tid in false_cands[:max_negatives_per_positive * max(len(true_targets), 1)]:
            t_row = t_lookup[tid]
            t_text = make_record_text(
                t_row.get("business_name", ""),
                t_row.get("business_address", ""),
                t_row.get("country", ""),
            )
            examples.append(InputExample(texts=[s1_text, t_text], label=0.0))

    random.shuffle(examples)
    print(f"Built {len(examples):,} training examples for bi-encoder fine-tuning.")
    return examples


def finetune_biencoder(
    model_name: str = "BAAI/bge-m3",
    examples: List["InputExample"] = None,
    output_dir: str = "models/biencoder_finetuned",
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
):
    """
    Fine-tune a bi-encoder with MultipleNegativesRankingLoss.
    Saves the fine-tuned model to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
    print(f"Training for {epochs} epochs, {len(train_dataloader)} batches/epoch, "
          f"warmup_steps={warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=output_dir,
        show_progress_bar=True,
    )

    print(f"Fine-tuned bi-encoder saved to {output_dir}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir
