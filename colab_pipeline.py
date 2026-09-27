"""
================================================================
 Business Entity Resolution — COMPLETE Colab Pipeline
 Amazon ML Challenge 2026
================================================================
 This script is designed to run on Google Colab with T4 GPU.
 Copy-paste each section (marked with # === CELL N ===) into
 a Colab notebook cell and run them sequentially.

 Architecture: Bi-Encoder Blocking → Cross-Encoder Reranking
               → Feature Stacking (LightGBM + Optuna)
               → Hungarian 1-to-1 Resolution
================================================================
"""

# === CELL 1: Install Dependencies ================================================
# !pip install -q sentence-transformers faiss-gpu lightgbm optuna rapidfuzz
# !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# === CELL 2: Mount Drive & Upload Data ==========================================
"""
from google.colab import drive
drive.mount('/content/drive')

# Upload your dataset folder to /content/drive/MyDrive/amazon_challenge/dataset/
# Expected structure:
# dataset/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
# dataset/test/test_source1.tsv, test_source2.tsv, test_source3.tsv

import os
BASE_DIR = '/content/drive/MyDrive/amazon_challenge'
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset/train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset/test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
"""

# === CELL 3: Core Utilities =====================================================
import os
import gc
import re
import json
import time
import random
import pickle
import unicodedata
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict

# ─── CONFIGURATION ───
# Set these to match your Colab drive paths
BASE_DIR = '.'  # Change to '/content/drive/MyDrive/amazon_challenge' on Colab
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset/train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset/test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BLOCKING_DIR = os.path.join(BASE_DIR, 'blocking_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BLOCKING_DIR, exist_ok=True)

# Model choices
BI_ENCODER_MODEL = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
TOP_K_BLOCKING = 50
BATCH_SIZE_EMBED = 512
BATCH_SIZE_CE = 128

# ─── NORMALIZATION ───

def strip_accents(text):
    if not text: return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_SUFFIX_RE = re.compile(
    r"\b(?:private limited|pvt limited|pvt ltd|private ltd|pvt\s+ltd|p\s+ltd|"
    r"pty\s+ltd|public limited company|public limited|limited liability company|"
    r"limited liability partnership|societe anonyme|"
    r"societe a responsabilite limitee|et fils|et freres|and sons|and brothers|"
    r"and co|and company|incorporated|incorporation|inc|corporation|corp|"
    r"limited|ltd|llc|llp|pllc|company|co|holdings|holding|group|"
    r"enterprises|enterprise|international|intl|pvt|private|"
    r"sarl|sasu|sas|sa|sci|eurl|snc|cie|groupe|fils|freres)\b",
    flags=re.IGNORECASE
)

_ADDR_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue",
    "blvd": "boulevard", "bvd": "boulevard", "bd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "sq": "square", "hwy": "highway", "bldg": "building",
    "fl": "floor", "apt": "apartment", "ste": "suite",
    "sec": "sector", "ext": "extension", "soc": "society",
    "col": "colony", "dist": "district", "nr": "near",
    "r": "rue", "imp": "impasse", "che": "chemin", "rte": "route",
}
_ADDR_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _ADDR_MAP) + r")\b", flags=re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+[a-z]?\b", flags=re.IGNORECASE)
_POSTAL_RE = re.compile(r"\b\d{5,6}\b")

def normalize_text(text):
    if text is None: return ""
    text = str(text)
    if not text or text.lower() == "nan": return ""
    text = strip_accents(text).lower()
    text = re.sub(r"&", " and ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()

def normalize_name(name):
    clean = normalize_text(name)
    if not clean: return ""
    clean = re.sub(r"\b(com|org|net|in|fr|co|gov|edu|io)\b", " ", clean)
    clean = _SUFFIX_RE.sub(" ", clean)
    return _MULTI_SPACE_RE.sub(" ", clean).strip() or normalize_text(name)

def normalize_address(address):
    clean = normalize_text(address)
    if not clean: return ""
    clean = _ADDR_RE.sub(lambda m: _ADDR_MAP[m.group(0).lower()], clean)
    return _MULTI_SPACE_RE.sub(" ", clean).strip()

def extract_numbers(text):
    return _NUM_RE.findall(str(text).lower()) if text else []

def extract_postal_code(text):
    if not text: return None
    m = _POSTAL_RE.findall(str(text))
    return m[-1] if m else None

def make_record_text(name, address, country):
    parts = []
    if name and str(name).lower() != "nan": parts.append(str(name).strip())
    if address and str(address).lower() != "nan": parts.append(str(address).strip())
    if country and str(country).lower() != "nan": parts.append(str(country).strip())
    return " | ".join(parts)

# ─── SCORING ───

def f05_single(pred, true):
    if not true: return 1.0 if not pred else 0.0
    if not pred: return 0.0
    tp = len(pred & true)
    if tp == 0: return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1.25 * p * r) / (0.25 * p + r)

def macro_f05(preds, truths, all_ids=None):
    if all_ids is None: all_ids = sorted(set(truths.keys()) | set(preds.keys()))
    scores = [f05_single(preds.get(sid, set()), truths.get(sid, set())) for sid in all_ids]
    return float(np.mean(scores))

# ─── FEATURES ───

from rapidfuzz import fuzz

def _soundex(token):
    if not token: return ""
    t = token.upper()
    first = t[0]
    m = {'B':'1','F':'1','P':'1','V':'1','C':'2','G':'2','J':'2','K':'2',
         'Q':'2','S':'2','X':'2','Z':'2','D':'3','T':'3','L':'4','M':'5','N':'5','R':'6'}
    d, prev = [], m.get(first, '0')
    for c in t[1:]:
        v = m.get(c, '0')
        if v != '0' and v != prev: d.append(v)
        prev = v
        if len(d) == 3: break
    return first + "".join(d).ljust(3, '0')[:3]

FEATURE_NAMES = [
    "name_token_sort_ratio", "name_token_set_ratio", "name_ratio",
    "name_partial_ratio", "name_compact_ratio", "name_jaccard",
    "name_common_tokens", "name_len_diff_ratio", "name_first_token_match",
    "name_soundex_match", "addr_is_null", "addr_token_sort_ratio",
    "addr_token_set_ratio", "addr_ratio", "addr_partial_ratio",
    "addr_jaccard", "addr_common_tokens", "num_exact_match", "num_jaccard",
    "num_count_s1", "num_count_t", "postal_match", "country_match",
    "target_source", "blocking_score", "candidate_rank", "total_candidates",
]

def compute_features(s1_name, s1_addr, s1_country, t_name, t_addr, t_country,
                     target_id, blocking_score=0.0, candidate_rank=0, total_candidates=0):
    cn1, cn2 = normalize_name(s1_name), normalize_name(t_name)
    ca1, ca2 = normalize_address(s1_addr), normalize_address(t_addr)
    comp1, comp2 = cn1.replace(" ", ""), cn2.replace(" ", "")

    nts = fuzz.token_sort_ratio(cn1, cn2) / 100.0
    ntset = fuzz.token_set_ratio(cn1, cn2) / 100.0
    nr = fuzz.ratio(cn1, cn2) / 100.0
    npr = fuzz.partial_ratio(cn1, cn2) / 100.0
    ncr = fuzz.ratio(comp1, comp2) / 100.0 if comp1 and comp2 else 0.0

    tok1, tok2 = set(cn1.split()), set(cn2.split())
    inter, union = tok1 & tok2, tok1 | tok2
    njac = len(inter) / len(union) if union else 0.0
    ncom = float(len(inter))
    ndiff = abs(len(cn1) - len(cn2)) / max(len(cn1), len(cn2), 1)

    ft1, ft2 = (cn1.split()[0] if cn1 else ""), (cn2.split()[0] if cn2 else "")
    nfm = 1.0 if ft1 and ft1 == ft2 else 0.0
    nsm = 1.0 if ft1 and _soundex(ft1) == _soundex(ft2) else 0.0

    a_null = 1.0 if (not ca1 or not ca2) else 0.0
    if a_null == 0.0:
        ats = fuzz.token_sort_ratio(ca1, ca2) / 100.0
        atset = fuzz.token_set_ratio(ca1, ca2) / 100.0
        ar = fuzz.ratio(ca1, ca2) / 100.0
        apr = fuzz.partial_ratio(ca1, ca2) / 100.0
        at1, at2 = set(ca1.split()), set(ca2.split())
        aint, aun = at1 & at2, at1 | at2
        ajac = len(aint) / len(aun) if aun else 0.0
        acom = float(len(aint))
    else:
        ats = atset = ar = apr = ajac = 0.0; acom = 0.0

    n1 = [x.lstrip('0') or '0' for x in extract_numbers(ca1)]
    n2 = [x.lstrip('0') or '0' for x in extract_numbers(ca2)]
    s1n, s2n = set(n1), set(n2)
    nex = 1.0 if s1n and s2n and (s1n & s2n) else 0.0
    nun = s1n | s2n
    njn = len(s1n & s2n) / len(nun) if nun else 0.0
    p1, p2 = extract_postal_code(ca1), extract_postal_code(ca2)
    pm = 1.0 if (p1 and p2 and p1 == p2) else (-1.0 if (p1 and p2) else 0.0)

    cm = 1.0 if s1_country and t_country and str(s1_country).strip().lower() == str(t_country).strip().lower() else 0.0
    ts = 2.0 if str(target_id).startswith("S2-") else 3.0

    return [nts, ntset, nr, npr, ncr, njac, ncom, ndiff, nfm, nsm,
            a_null, ats, atset, ar, apr, ajac, acom,
            nex, njn, float(len(s1n)), float(len(s2n)), pm,
            cm, ts, blocking_score, float(candidate_rank), float(total_candidates)]

print("✅ Core utilities loaded.")


# === CELL 4: Load Data ===========================================================

print("Loading training data...")
train_s1 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t')
train_s2 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t')
train_s3 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t')
train_gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t')
train_targets = pd.concat([train_s2, train_s3], ignore_index=True)

# Build ground truth map
gt_map = {}
for _, row in train_gt.iterrows():
    sid = row['source1_entity_id']
    t_str = str(row.get('matched_entity_ids', ''))
    if t_str and t_str.lower() != 'nan' and t_str.strip():
        gt_map[sid] = set(t_str.split(','))
    else:
        gt_map[sid] = set()

# Train/val split — entity-level (80/20)
all_s1_ids = train_s1['entity_id'].tolist()
random.seed(42)
random.shuffle(all_s1_ids)
split_idx = int(0.8 * len(all_s1_ids))
train_ids = set(all_s1_ids[:split_idx])
val_ids = set(all_s1_ids[split_idx:])

train_s1_train = train_s1[train_s1['entity_id'].isin(train_ids)].reset_index(drop=True)
train_s1_val = train_s1[train_s1['entity_id'].isin(val_ids)].reset_index(drop=True)

print(f"Train S1: {len(train_s1):,} | S2: {len(train_s2):,} | S3: {len(train_s3):,}")
print(f"Ground truth entries: {len(gt_map):,}")
print(f"Train split: {len(train_s1_train):,} | Val split: {len(train_s1_val):,}")
print(f"Countries: {train_s1['country'].unique()}")
print("✅ Data loaded.")


# === CELL 5: Bi-Encoder Blocking =================================================

import torch
import faiss
from sentence_transformers import SentenceTransformer

def build_texts_from_df(df):
    return [make_record_text(
        row.get("business_name", ""),
        row.get("business_address", ""),
        row.get("country", ""),
    ) for _, row in df.iterrows()]

def encode_batched(model, texts, batch_size=512, normalize_embeddings=True):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=normalize_embeddings,
                        convert_to_numpy=True).astype(np.float32)

print("Loading bi-encoder model...")
bi_model = SentenceTransformer(BI_ENCODER_MODEL)
if torch.cuda.is_available():
    bi_model = bi_model.to(torch.device("cuda"))
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Embed S1 (train split for validation)
print("Encoding S1 train records...")
s1_train_texts = build_texts_from_df(train_s1_train)
s1_train_emb = encode_batched(bi_model, s1_train_texts, batch_size=BATCH_SIZE_EMBED)

print("Encoding S1 val records...")
s1_val_texts = build_texts_from_df(train_s1_val)
s1_val_emb = encode_batched(bi_model, s1_val_texts, batch_size=BATCH_SIZE_EMBED)

# Embed all targets (S2 + S3)
print("Encoding target records (S2 + S3)...")
target_texts = build_texts_from_df(train_targets)
target_emb = encode_batched(bi_model, target_texts, batch_size=BATCH_SIZE_EMBED)

# Save embeddings
np.save(os.path.join(BLOCKING_DIR, 's1_train_emb.npy'), s1_train_emb)
np.save(os.path.join(BLOCKING_DIR, 's1_val_emb.npy'), s1_val_emb)
np.save(os.path.join(BLOCKING_DIR, 'target_emb.npy'), target_emb)
print("✅ Embeddings saved.")

# FAISS retrieval
target_ids = train_targets['entity_id'].tolist()
target_countries = train_targets['country'].fillna("").tolist()

dim = target_emb.shape[1]
index = faiss.IndexFlatIP(dim)
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(target_emb)
    search_index = gpu_index
else:
    index.add(target_emb)
    search_index = index

def retrieve_candidates_for_split(s1_emb, s1_df, top_k):
    s1_ids = s1_df['entity_id'].tolist()
    s1_countries = s1_df['country'].fillna("").tolist()
    D, I = search_index.search(s1_emb, top_k)
    candidates = {}
    for i, s1_id in enumerate(s1_ids):
        s1_c = s1_countries[i].strip().lower()
        cands = []
        for j in range(top_k):
            idx = int(I[i, j])
            if idx < 0: continue
            tid = target_ids[idx]
            t_c = target_countries[idx].strip().lower()
            score = float(D[i, j])
            # Country filter — only skip if both have a country and they differ
            if s1_c and t_c and s1_c != t_c:
                continue
            cands.append((tid, score))
        candidates[s1_id] = cands
    return candidates

print(f"Retrieving top-{TOP_K_BLOCKING} candidates for train split...")
cands_train = retrieve_candidates_for_split(s1_train_emb, train_s1_train, TOP_K_BLOCKING)

print(f"Retrieving top-{TOP_K_BLOCKING} candidates for val split...")
cands_val = retrieve_candidates_for_split(s1_val_emb, train_s1_val, TOP_K_BLOCKING)

# Measure recall ceiling
def recall_ceiling(candidates, gt):
    total = retrieved = 0
    for s1_id, true_set in gt.items():
        if not true_set or s1_id not in candidates: continue
        total += len(true_set)
        cand_ids = {c[0] for c in candidates.get(s1_id, [])}
        retrieved += len(true_set & cand_ids)
    return retrieved / total if total else 1.0, retrieved, total

rc_train, ret_train, tot_train = recall_ceiling(cands_train, gt_map)
rc_val, ret_val, tot_val = recall_ceiling(cands_val, gt_map)
print(f"\n📊 Blocking Recall Ceiling:")
print(f"  Train: {rc_train:.4f} ({ret_train}/{tot_train})")
print(f"  Val:   {rc_val:.4f} ({ret_val}/{tot_val})")

# Free GPU memory
del bi_model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("✅ Blocking complete.")


# === CELL 6: Fine-tune Bi-Encoder (OPTIONAL — skip if recall ceiling > 0.98) =====
"""
# Uncomment this cell if blocking recall < 0.98 and you need to improve it.

from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

t_lookup = {row["entity_id"]: row for _, row in train_targets.iterrows()}

# Build contrastive training pairs
examples = []
for s1_id in train_ids:
    if s1_id not in gt_map: continue
    s1_row = train_s1[train_s1['entity_id'] == s1_id].iloc[0]
    s1_text = make_record_text(s1_row['business_name'], s1_row['business_address'], s1_row['country'])
    true_set = gt_map[s1_id]

    for tid in true_set:
        if tid in t_lookup:
            t_row = t_lookup[tid]
            t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
            examples.append(InputExample(texts=[s1_text, t_text], label=1.0))

    # Hard negatives from blocking
    false_cands = [c[0] for c in cands_train.get(s1_id, []) if c[0] not in true_set and c[0] in t_lookup]
    random.shuffle(false_cands)
    for tid in false_cands[:3 * max(len(true_set), 1)]:
        t_row = t_lookup[tid]
        t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
        examples.append(InputExample(texts=[s1_text, t_text], label=0.0))

random.shuffle(examples)
print(f"Built {len(examples):,} fine-tuning examples.")

ft_model = SentenceTransformer(BI_ENCODER_MODEL)
train_dl = DataLoader(examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(ft_model)
ft_model.fit(
    train_objectives=[(train_dl, train_loss)],
    epochs=3, warmup_steps=int(len(train_dl)*0.1),
    output_path=os.path.join(MODELS_DIR, 'biencoder_finetuned'),
    show_progress_bar=True
)
print("✅ Bi-encoder fine-tuned and saved.")
# After fine-tuning, re-run blocking with the fine-tuned model and re-check recall.
"""


# === CELL 7: Cross-Encoder Re-Ranking ============================================

from sentence_transformers import CrossEncoder

print("Loading cross-encoder model...")
ce_model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=256)

t_lookup = {row["entity_id"]: row for _, row in train_targets.iterrows()}

def score_candidates_ce(candidates_dict, s1_df, batch_size=BATCH_SIZE_CE):
    """Score all candidate pairs with cross-encoder."""
    s1_lookup = {row["entity_id"]: row for _, row in s1_df.iterrows()}
    all_pairs = []
    all_keys = []

    for s1_id, cand_list in candidates_dict.items():
        if s1_id not in s1_lookup: continue
        s1_row = s1_lookup[s1_id]
        s1_text = make_record_text(s1_row['business_name'], s1_row['business_address'], s1_row['country'])
        for tid, _ in cand_list:
            if tid not in t_lookup: continue
            t_row = t_lookup[tid]
            t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
            all_pairs.append([s1_text, t_text])
            all_keys.append((s1_id, tid))

    print(f"Scoring {len(all_pairs):,} pairs with cross-encoder...")
    if not all_pairs: return {}

    scores = ce_model.predict(all_pairs, batch_size=batch_size, show_progress_bar=True)
    scores = 1.0 / (1.0 + np.exp(-np.array(scores)))  # sigmoid

    ce_scores = {}
    for (s1_id, tid), score in zip(all_keys, scores):
        if s1_id not in ce_scores: ce_scores[s1_id] = []
        ce_scores[s1_id].append((tid, float(score)))

    for s1_id in ce_scores:
        ce_scores[s1_id].sort(key=lambda x: x[1], reverse=True)
    return ce_scores

print("Scoring TRAIN candidates...")
ce_scores_train = score_candidates_ce(cands_train, train_s1_train)

print("Scoring VAL candidates...")
ce_scores_val = score_candidates_ce(cands_val, train_s1_val)

# Quick sanity check — CE-only baseline
def ce_only_baseline(ce_scores, gt, threshold=0.5):
    preds = {}
    for s1_id, scored in ce_scores.items():
        preds[s1_id] = {tid for tid, s in scored if s >= threshold}
    return macro_f05(preds, gt, all_ids=list(ce_scores.keys()))

for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    score = ce_only_baseline(ce_scores_val, gt_map, threshold=th)
    print(f"  CE-only baseline (threshold={th}): F0.5 = {score:.4f}")

del ce_model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("✅ Cross-encoder scoring done.")


# === CELL 8: Fine-tune Cross-Encoder =============================================

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# Build training pairs from TRAIN split
ce_examples = []
for s1_id in train_ids:
    if s1_id not in gt_map: continue
    s1_row_match = train_s1[train_s1['entity_id'] == s1_id]
    if s1_row_match.empty: continue
    s1_row = s1_row_match.iloc[0]
    s1_text = make_record_text(s1_row['business_name'], s1_row['business_address'], s1_row['country'])
    true_set = gt_map[s1_id]

    for tid in true_set:
        if tid in t_lookup:
            t_row = t_lookup[tid]
            t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
            ce_examples.append(InputExample(texts=[s1_text, t_text], label=1))

    false_cands = [c[0] for c in cands_train.get(s1_id, []) if c[0] not in true_set and c[0] in t_lookup]
    random.shuffle(false_cands)
    for tid in false_cands[:5 * max(len(true_set), 1)]:
        t_row = t_lookup[tid]
        t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
        ce_examples.append(InputExample(texts=[s1_text, t_text], label=0))

random.shuffle(ce_examples)
print(f"Cross-encoder fine-tuning: {len(ce_examples):,} examples")

ce_ft_model = CrossEncoder(CROSS_ENCODER_MODEL, num_labels=1, max_length=256)
ce_ft_dl = DataLoader(ce_examples, shuffle=True, batch_size=32)
warmup = int(len(ce_ft_dl) * 3 * 0.1)

ce_ft_model.fit(
    train_dataloader=ce_ft_dl,
    epochs=3,
    warmup_steps=warmup,
    output_path=os.path.join(MODELS_DIR, 'cross_encoder_finetuned'),
    show_progress_bar=True,
)
print("✅ Cross-encoder fine-tuned.")

# Re-score val candidates with fine-tuned CE
print("Re-scoring VAL with fine-tuned CE...")
ce_ft_model_reload = CrossEncoder(os.path.join(MODELS_DIR, 'cross_encoder_finetuned'), max_length=256)

s1_val_lookup = {row["entity_id"]: row for _, row in train_s1_val.iterrows()}
ft_pairs, ft_keys = [], []
for s1_id, cand_list in cands_val.items():
    if s1_id not in s1_val_lookup: continue
    s1_row = s1_val_lookup[s1_id]
    s1_text = make_record_text(s1_row['business_name'], s1_row['business_address'], s1_row['country'])
    for tid, _ in cand_list:
        if tid not in t_lookup: continue
        t_row = t_lookup[tid]
        t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
        ft_pairs.append([s1_text, t_text])
        ft_keys.append((s1_id, tid))

ft_scores_raw = ce_ft_model_reload.predict(ft_pairs, batch_size=BATCH_SIZE_CE, show_progress_bar=True)
ft_scores_sig = 1.0 / (1.0 + np.exp(-np.array(ft_scores_raw)))

ce_scores_val_ft = {}
for (s1_id, tid), score in zip(ft_keys, ft_scores_sig):
    if s1_id not in ce_scores_val_ft: ce_scores_val_ft[s1_id] = []
    ce_scores_val_ft[s1_id].append((tid, float(score)))

for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    score = ce_only_baseline(ce_scores_val_ft, gt_map, threshold=th)
    print(f"  Fine-tuned CE baseline (threshold={th}): F0.5 = {score:.4f}")

# Re-score TRAIN with fine-tuned CE for GBM training
print("Re-scoring TRAIN with fine-tuned CE...")
s1_train_lookup = {row["entity_id"]: row for _, row in train_s1_train.iterrows()}
ft_train_pairs, ft_train_keys = [], []
for s1_id, cand_list in cands_train.items():
    if s1_id not in s1_train_lookup: continue
    s1_row = s1_train_lookup[s1_id]
    s1_text = make_record_text(s1_row['business_name'], s1_row['business_address'], s1_row['country'])
    for tid, _ in cand_list:
        if tid not in t_lookup: continue
        t_row = t_lookup[tid]
        t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
        ft_train_pairs.append([s1_text, t_text])
        ft_train_keys.append((s1_id, tid))

ft_train_scores_raw = ce_ft_model_reload.predict(ft_train_pairs, batch_size=BATCH_SIZE_CE, show_progress_bar=True)
ft_train_scores_sig = 1.0 / (1.0 + np.exp(-np.array(ft_train_scores_raw)))

ce_scores_train_ft = {}
for (s1_id, tid), score in zip(ft_train_keys, ft_train_scores_sig):
    if s1_id not in ce_scores_train_ft: ce_scores_train_ft[s1_id] = []
    ce_scores_train_ft[s1_id].append((tid, float(score)))

del ce_ft_model, ce_ft_model_reload
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("✅ Fine-tuned CE scoring complete.")


# === CELL 9: Build Feature Matrix + GBM Training with Optuna =====================

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def build_feature_matrix(candidates_dict, ce_scores_dict, s1_df, gt):
    s1_lookup = {row["entity_id"]: row for _, row in s1_df.iterrows()}
    ce_lookup = {}
    for s1_id, scores in ce_scores_dict.items():
        for tid, score in scores:
            ce_lookup[(s1_id, tid)] = score

    X_list, y_list, s1_list, t_list = [], [], [], []
    for s1_id, cand_list in candidates_dict.items():
        if s1_id not in s1_lookup: continue
        s1_row = s1_lookup[s1_id]
        true_set = gt.get(s1_id, set())
        n_cands = len(cand_list)
        for rank, (tid, bscore) in enumerate(cand_list):
            if tid not in t_lookup: continue
            t_row = t_lookup[tid]
            feats = compute_features(
                s1_row['business_name'], s1_row['business_address'], s1_row['country'],
                t_row['business_name'], t_row['business_address'], t_row['country'],
                tid, bscore, rank, n_cands,
            )
            feats.append(ce_lookup.get((s1_id, tid), 0.0))  # cross-encoder score
            X_list.append(feats)
            y_list.append(1 if tid in true_set else 0)
            s1_list.append(s1_id)
            t_list.append(tid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Feature matrix: {X.shape[0]:,} pairs, {y.sum():,} positives ({100*y.mean():.2f}%)")
    return X, y, s1_list, t_list

print("Building TRAIN feature matrix...")
X_train, y_train, s1_ids_train, t_ids_train = build_feature_matrix(
    cands_train, ce_scores_train_ft, train_s1_train, gt_map)

print("Building VAL feature matrix...")
X_val, y_val, s1_ids_val, t_ids_val = build_feature_matrix(
    cands_val, ce_scores_val_ft, train_s1_val, gt_map)

# Hungarian 1-to-1 resolution
def hungarian_resolve(s1_ids, t_ids, probs, threshold, max_matches=11):
    all_triples = []
    for s1, tid, p in zip(s1_ids, t_ids, probs):
        if p >= threshold:
            all_triples.append((p, s1, tid))
    all_triples.sort(reverse=True)

    accepted = defaultdict(set)
    claimed = set()
    counts = defaultdict(int)
    for prob, s1, tid in all_triples:
        if tid in claimed: continue
        if counts[s1] >= max_matches: continue
        accepted[s1].add(tid)
        claimed.add(tid)
        counts[s1] += 1
    return dict(accepted)

feature_names = FEATURE_NAMES + ["cross_encoder_score"]

def optuna_objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
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

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    full_params = {"objective": "binary", "metric": "binary_logloss",
                   "verbosity": -1, "num_threads": -1,
                   "scale_pos_weight": neg / max(pos, 1), **params}

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)
    model = lgb.train(full_params, dtrain, num_boost_round=500,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])

    probs = model.predict(X_val)
    preds = hungarian_resolve(s1_ids_val, t_ids_val, probs, threshold, max_matches)
    return macro_f05(preds, gt_map, all_ids=sorted(set(s1_ids_val)))

print("Running Optuna optimization (100 trials)...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_objective, n_trials=100, show_progress_bar=True)

best = study.best_trial
print(f"\n{'='*60}")
print(f"🏆 Best trial #{best.number}: macro F0.5 = {best.value:.5f}")
print(f"Best params: {json.dumps(best.params, indent=2)}")
print(f"{'='*60}")

# Save best params
with open(os.path.join(MODELS_DIR, 'optuna_best.json'), 'w') as f:
    json.dump({"best_params": best.params, "best_score": best.value}, f, indent=2)
print("✅ Optuna tuning complete.")


# === CELL 10: Train Final Model on ALL Training Data =============================

best_params = study.best_params.copy()
best_threshold = best_params.pop("threshold")
best_max_matches = best_params.pop("max_matches")
best_lr = best_params.pop("lr")
best_params["learning_rate"] = best_lr

# Combine train + val for final model
X_all = np.vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])

neg, pos = (y_all == 0).sum(), (y_all == 1).sum()
full_params = {"objective": "binary", "metric": "binary_logloss",
               "verbosity": -1, "num_threads": -1,
               "scale_pos_weight": neg / max(pos, 1), **best_params}

dtrain_all = lgb.Dataset(X_all, label=y_all, feature_name=feature_names)
final_model = lgb.train(full_params, dtrain_all, num_boost_round=800)
final_model.save_model(os.path.join(MODELS_DIR, 'lgbm_final.txt'))

print(f"Final model trained on {len(X_all):,} pairs.")
print(f"Decision config: threshold={best_threshold:.4f}, max_matches={best_max_matches}")

# Save config
with open(os.path.join(MODELS_DIR, 'decision_config.json'), 'w') as f:
    json.dump({"threshold": best_threshold, "max_matches": int(best_max_matches)}, f)
print("✅ Final model trained and saved.")


# === CELL 11: TEST SET INFERENCE =================================================

print("\n" + "="*60)
print("🚀 STARTING TEST SET INFERENCE")
print("="*60)

# Load test data
print("Loading test data...")
test_s1 = pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t')
test_s2 = pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t')
test_s3 = pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t')
test_targets = pd.concat([test_s2, test_s3], ignore_index=True)
print(f"Test S1: {len(test_s1):,} | S2: {len(test_s2):,} | S3: {len(test_s3):,}")
print(f"Countries: {test_s1['country'].unique()}")

# Load fine-tuned bi-encoder for test blocking
print("\nStep 1: Bi-encoder blocking on test set...")
bi_model_path = os.path.join(MODELS_DIR, 'biencoder_finetuned')
if os.path.exists(bi_model_path):
    bi_model_test = SentenceTransformer(bi_model_path)
    print("Using fine-tuned bi-encoder.")
else:
    bi_model_test = SentenceTransformer(BI_ENCODER_MODEL)
    print("Using base bi-encoder (no fine-tuned model found).")
if torch.cuda.is_available():
    bi_model_test = bi_model_test.to(torch.device("cuda"))

test_s1_texts = build_texts_from_df(test_s1)
test_target_texts = build_texts_from_df(test_targets)

print(f"Encoding {len(test_s1_texts):,} test S1 records...")
test_s1_emb = encode_batched(bi_model_test, test_s1_texts, batch_size=BATCH_SIZE_EMBED)

print(f"Encoding {len(test_target_texts):,} test target records...")
test_target_emb = encode_batched(bi_model_test, test_target_texts, batch_size=BATCH_SIZE_EMBED)

# FAISS index for test
dim = test_target_emb.shape[1]
test_index = faiss.IndexFlatIP(dim)
if faiss.get_num_gpus() > 0:
    test_res = faiss.StandardGpuResources()
    test_gpu_index = faiss.index_cpu_to_gpu(test_res, 0, test_index)
    test_gpu_index.add(test_target_emb)
    test_search = test_gpu_index
else:
    test_index.add(test_target_emb)
    test_search = test_index

test_s1_ids = test_s1['entity_id'].tolist()
test_s1_countries = test_s1['country'].fillna("").tolist()
test_target_ids = test_targets['entity_id'].tolist()
test_target_countries = test_targets['country'].fillna("").tolist()

print(f"FAISS search: top-{TOP_K_BLOCKING}...")
D_test, I_test = test_search.search(test_s1_emb, TOP_K_BLOCKING)

test_candidates = {}
for i, s1_id in enumerate(test_s1_ids):
    s1_c = test_s1_countries[i].strip().lower()
    cands = []
    for j in range(TOP_K_BLOCKING):
        idx = int(I_test[i, j])
        if idx < 0: continue
        tid = test_target_ids[idx]
        t_c = test_target_countries[idx].strip().lower()
        score = float(D_test[i, j])
        if s1_c and t_c and s1_c != t_c: continue
        cands.append((tid, score))
    test_candidates[s1_id] = cands

n_with_cands = sum(1 for c in test_candidates.values() if c)
print(f"Blocking: {n_with_cands:,}/{len(test_s1_ids):,} S1 entities have candidates.")

del bi_model_test, test_s1_emb, test_target_emb
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Cross-encoder scoring on test
print("\nStep 2: Cross-encoder re-ranking on test set...")
ce_model_path = os.path.join(MODELS_DIR, 'cross_encoder_finetuned')
if os.path.exists(ce_model_path):
    ce_test_model = CrossEncoder(ce_model_path, max_length=256)
    print("Using fine-tuned cross-encoder.")
else:
    ce_test_model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=256)
    print("Using base cross-encoder.")

test_t_lookup = {row["entity_id"]: row for _, row in test_targets.iterrows()}
test_s1_lookup = {row["entity_id"]: row for _, row in test_s1.iterrows()}

ce_test_pairs, ce_test_keys = [], []
for s1_id, cand_list in test_candidates.items():
    if s1_id not in test_s1_lookup: continue
    s1_row = test_s1_lookup[s1_id]
    s1_text = make_record_text(s1_row['business_name'], s1_row['business_address'], s1_row['country'])
    for tid, _ in cand_list:
        if tid not in test_t_lookup: continue
        t_row = test_t_lookup[tid]
        t_text = make_record_text(t_row['business_name'], t_row['business_address'], t_row['country'])
        ce_test_pairs.append([s1_text, t_text])
        ce_test_keys.append((s1_id, tid))

print(f"Scoring {len(ce_test_pairs):,} test candidate pairs...")
ce_test_scores_raw = ce_test_model.predict(ce_test_pairs, batch_size=BATCH_SIZE_CE, show_progress_bar=True)
ce_test_scores = 1.0 / (1.0 + np.exp(-np.array(ce_test_scores_raw)))

ce_test_score_lookup = {}
for (s1_id, tid), score in zip(ce_test_keys, ce_test_scores):
    ce_test_score_lookup[(s1_id, tid)] = float(score)

del ce_test_model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Build test features
print("\nStep 3: Building test feature matrix...")
X_test_list, test_pair_s1, test_pair_t = [], [], []
for s1_id, cand_list in test_candidates.items():
    if s1_id not in test_s1_lookup: continue
    s1_row = test_s1_lookup[s1_id]
    n_cands = len(cand_list)
    for rank, (tid, bscore) in enumerate(cand_list):
        if tid not in test_t_lookup: continue
        t_row = test_t_lookup[tid]
        feats = compute_features(
            s1_row['business_name'], s1_row['business_address'], s1_row['country'],
            t_row['business_name'], t_row['business_address'], t_row['country'],
            tid, bscore, rank, n_cands,
        )
        feats.append(ce_test_score_lookup.get((s1_id, tid), 0.0))
        X_test_list.append(feats)
        test_pair_s1.append(s1_id)
        test_pair_t.append(tid)

X_test = np.array(X_test_list, dtype=np.float32)
print(f"Test feature matrix: {X_test.shape}")

# Predict
print("\nStep 4: GBM prediction + Hungarian resolution...")
test_probs = final_model.predict(X_test)
test_matches = hungarian_resolve(test_pair_s1, test_pair_t, test_probs,
                                  best_threshold, int(best_max_matches))

# Write output
print("\nStep 5: Writing output files...")

# matching_results.tsv
matching_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
with open(matching_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in test_s1_ids:
        match_set = test_matches.get(s1_id, set())
        match_str = ",".join(sorted(match_set))
        f.write(f"{s1_id}\t{match_str}\n")

# candidate_pairs.tsv
candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
with open(candidate_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in test_s1_ids:
        cand_ids = [c[0] for c in test_candidates.get(s1_id, [])]
        f.write(f"{s1_id}\t{','.join(cand_ids)}\n")

n_matched = sum(1 for s in test_matches.values() if s)
n_empty = len(test_s1_ids) - n_matched
print(f"\n📊 Inference Results:")
print(f"  Total S1 entities: {len(test_s1_ids):,}")
print(f"  Matched entities:  {n_matched:,}")
print(f"  Singletons:        {n_empty:,}")
print(f"  Total matches:     {sum(len(s) for s in test_matches.values()):,}")
print(f"\n  Files saved:")
print(f"    {matching_path}")
print(f"    {candidate_path}")
print(f"\n✅ INFERENCE COMPLETE — matching_results.tsv ready for leaderboard!")
