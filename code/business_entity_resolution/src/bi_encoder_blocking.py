"""
Bi-Encoder Blocking — embed every record with a multilingual sentence
model, index with FAISS, retrieve top-K candidates per S1 entity.

Designed for GPU execution on Google Colab.
"""
import os
import gc
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# GPU libs — will be installed on Colab
try:
    import torch
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass  # handled at runtime

from normalize import make_record_text


def build_texts(df: pd.DataFrame) -> List[str]:
    """Create combined text for each record."""
    texts = []
    for _, row in df.iterrows():
        texts.append(make_record_text(
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", ""),
        ))
    return texts


def encode_batched(
    model: "SentenceTransformer",
    texts: List[str],
    batch_size: int = 512,
    show_progress: bool = True,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Encode texts in batches, returning L2-normalised float32 embeddings."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    ).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> "faiss.Index":
    """Build a FAISS inner-product index (cosine sim on L2-normalised vecs)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)           # inner product = cosine on normalised
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings)
    return index


def retrieve_candidates(
    s1_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    s1_ids: List[str],
    target_ids: List[str],
    s1_countries: List[str],
    target_countries: List[str],
    top_k: int = 50,
    use_gpu: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Retrieve top-K candidates for each S1 entity using FAISS.
    Applies soft country filtering post-retrieval.
    Returns: {s1_id: [(target_id, cosine_score), ...]}
    """
    print(f"Building FAISS index over {len(target_ids):,} targets...")
    index = build_faiss_index(target_embeddings, use_gpu=use_gpu)

    print(f"Querying {len(s1_ids):,} S1 entities (top_k={top_k})...")
    # Query in batches to avoid OOM
    batch_size = 4096
    all_scores = []
    all_indices = []
    for start in range(0, len(s1_ids), batch_size):
        end = min(start + batch_size, len(s1_ids))
        q = s1_embeddings[start:end]
        scores, indices = index.search(q, top_k)
        all_scores.append(scores)
        all_indices.append(indices)
    D = np.vstack(all_scores)
    I = np.vstack(all_indices)

    candidates = {}
    for i, s1_id in enumerate(s1_ids):
        s1_country = s1_countries[i].strip().lower()
        cands = []
        for j in range(top_k):
            idx = int(I[i, j])
            if idx < 0:
                continue
            tid = target_ids[idx]
            t_country = target_countries[idx].strip().lower()
            score = float(D[i, j])
            # Soft country filter: only exclude if BOTH have a country AND they differ
            # AND the similarity is already weak (< 0.5)
            if s1_country and t_country and s1_country != t_country:
                if score < 0.5:
                    continue
            cands.append((tid, score))
        candidates[s1_id] = cands
    return candidates


def compute_recall_ceiling(
    candidates: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, set],
) -> Tuple[float, int, int]:
    """Fraction of true matches surviving in the candidate set."""
    total = retrieved = 0
    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue
        total += len(true_set)
        cand_ids = {c[0] for c in candidates.get(s1_id, [])}
        retrieved += len(true_set & cand_ids)
    recall = retrieved / total if total else 1.0
    return recall, retrieved, total


def run_blocking(
    model_name: str,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = 50,
    batch_size: int = 512,
    use_gpu: bool = True,
    save_dir: str = "blocking_output",
) -> Dict[str, List[Tuple[str, float]]]:
    """
    End-to-end blocking: load model, embed, index, retrieve.
    Saves embeddings to disk for reuse.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading bi-encoder model: {model_name}")
    model = SentenceTransformer(model_name)
    if use_gpu and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    # Build texts
    print("Building S1 texts...")
    s1_texts = build_texts(s1_df)
    print("Building target texts...")
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    target_texts = build_texts(targets_df)

    # Encode
    print(f"Encoding {len(s1_texts):,} S1 records...")
    s1_emb = encode_batched(model, s1_texts, batch_size=batch_size)
    np.save(os.path.join(save_dir, "s1_embeddings.npy"), s1_emb)

    print(f"Encoding {len(target_texts):,} target records...")
    target_emb = encode_batched(model, target_texts, batch_size=batch_size)
    np.save(os.path.join(save_dir, "target_embeddings.npy"), target_emb)

    # IDs and countries
    s1_ids = s1_df["entity_id"].tolist()
    target_ids = targets_df["entity_id"].tolist()
    s1_countries = s1_df["country"].fillna("").tolist()
    target_countries = targets_df["country"].fillna("").tolist()

    # Retrieve
    candidates = retrieve_candidates(
        s1_emb, target_emb,
        s1_ids, target_ids,
        s1_countries, target_countries,
        top_k=top_k, use_gpu=use_gpu,
    )

    # Save candidate map
    cand_save = {k: [(t, float(s)) for t, s in v] for k, v in candidates.items()}
    with open(os.path.join(save_dir, "candidates.json"), "w") as f:
        json.dump(cand_save, f)

    # Free GPU memory
    del model, s1_emb, target_emb
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return candidates
