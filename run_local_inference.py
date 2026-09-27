"""
LOCAL Inference Pipeline — CPU only.
Uses the existing inverted-index blocking + LightGBM pipeline to produce
matching_results.tsv immediately for leaderboard submission.
Runs on the user's local machine (no GPU needed).
"""

import os
import sys
import time
import re
import unicodedata
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np

# Polars for efficient data loading
import polars as pl
from rapidfuzz import fuzz

# ─── Inline normalization (self-contained, no imports needed) ───

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


def get_token_set(text):
    return set(text.split()) if text else set()


def clean_num(n):
    return n.lstrip('0') or '0'


def soundex(token):
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


# ─── Blocking index ───

class SimpleBlockingIndex:
    """Fast inverted-index blocking with 5 channels."""
    def __init__(self):
        self.name_tok_idx = defaultdict(set)     # normalized name tokens
        self.soundex_idx = defaultdict(set)       # soundex codes
        self.compact_idx = defaultdict(set)       # 3-char compact name ngrams
        self.num_idx = defaultdict(set)           # address numbers
        self.postal_idx = defaultdict(set)        # postal codes
        self.target_data = {}

    def add_target(self, tid, name, addr, country):
        cn = normalize_name(name)
        ca = normalize_address(addr) if addr and str(addr).lower() != "nan" else ""
        compact = cn.replace(" ", "")
        nums = [clean_num(n) for n in extract_numbers(ca)]
        postal = extract_postal_code(ca)

        self.target_data[tid] = (cn, compact, ca, nums, postal, country)

        for tok in cn.split():
            if len(tok) >= 2:
                self.name_tok_idx[tok].add(tid)
                sx = soundex(tok)
                if sx: self.soundex_idx[sx].add(tid)

        for i in range(max(0, len(compact) - 2)):
            self.compact_idx[compact[i:i+3]].add(tid)

        for n in nums:
            self.num_idx[n].add(tid)
        if postal:
            self.postal_idx[postal].add(tid)

    def get_candidates(self, s1_id, name, addr, country, max_cands=30, min_floor=24.0):
        cn = normalize_name(name)
        ca = normalize_address(addr) if addr and str(addr).lower() != "nan" else ""
        compact = cn.replace(" ", "")
        nums = [clean_num(n) for n in extract_numbers(ca)]
        postal = extract_postal_code(ca)

        # Gather candidates from all channels
        scores = defaultdict(float)
        name_toks = cn.split()
        for tok in name_toks:
            if len(tok) >= 2:
                for tid in self.name_tok_idx.get(tok, set()):
                    scores[tid] += 10.0
                sx = soundex(tok)
                if sx:
                    for tid in self.soundex_idx.get(sx, set()):
                        scores[tid] += 3.0

        for i in range(max(0, len(compact) - 2)):
            trigram = compact[i:i+3]
            for tid in self.compact_idx.get(trigram, set()):
                scores[tid] += 2.0

        for n in nums:
            for tid in self.num_idx.get(n, set()):
                scores[tid] += 8.0
        if postal:
            for tid in self.postal_idx.get(postal, set()):
                scores[tid] += 15.0

        # Remove self
        scores.pop(s1_id, None)

        # Sort and filter
        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        filtered = [(tid, sc) for tid, sc in sorted_cands if sc >= min_floor]
        return filtered[:max_cands]


# ─── Feature computation ───

def compute_pairwise_features(s1_cn, s1_compact, s1_ca, s1_nums, s1_postal, s1_country,
                               target_id, t_cn, t_compact, t_ca, t_nums, t_postal, t_country,
                               coarse_score, cand_rank, top_score, total_cands):
    """28-dimensional feature vector."""
    # Name features
    nts = fuzz.token_sort_ratio(s1_cn, t_cn) / 100.0
    ntset = fuzz.token_set_ratio(s1_cn, t_cn) / 100.0
    nr = fuzz.ratio(s1_cn, t_cn) / 100.0
    npr = fuzz.partial_ratio(s1_cn, t_cn) / 100.0
    ncr = fuzz.ratio(s1_compact, t_compact) / 100.0 if s1_compact and t_compact else 0.0

    tok1, tok2 = set(s1_cn.split()), set(t_cn.split())
    inter = tok1 & tok2
    union = tok1 | tok2
    njac = len(inter) / len(union) if union else 0.0
    ncom = float(len(inter))
    ndiff = abs(len(s1_cn) - len(t_cn)) / max(len(s1_cn), len(t_cn), 1)

    ft1 = s1_cn.split()[0] if s1_cn else ""
    ft2 = t_cn.split()[0] if t_cn else ""
    nfm = 1.0 if ft1 and ft1 == ft2 else 0.0
    nsm = 1.0 if ft1 and soundex(ft1) == soundex(ft2) else 0.0

    # Address features
    a_null = 1.0 if (not s1_ca or not t_ca) else 0.0
    if a_null == 0.0:
        ats = fuzz.token_sort_ratio(s1_ca, t_ca) / 100.0
        atset = fuzz.token_set_ratio(s1_ca, t_ca) / 100.0
        ar = fuzz.ratio(s1_ca, t_ca) / 100.0
        apr = fuzz.partial_ratio(s1_ca, t_ca) / 100.0
        at1 = set(s1_ca.split()); at2 = set(t_ca.split())
        aint = at1 & at2; aun = at1 | at2
        ajac = len(aint) / len(aun) if aun else 0.0
        acom = float(len(aint))
    else:
        ats = atset = ar = apr = ajac = 0.0; acom = 0.0

    # Number features
    s1n, s2n = set(s1_nums), set(t_nums)
    nex = 1.0 if s1n and s2n and (s1n & s2n) else 0.0
    nun = s1n | s2n
    njn = len(s1n & s2n) / len(nun) if nun else 0.0
    pm = 1.0 if (s1_postal and t_postal and s1_postal == t_postal) else (-1.0 if (s1_postal and t_postal) else 0.0)

    # Structural
    cm = 1.0 if s1_country and t_country and str(s1_country).strip().lower() == str(t_country).strip().lower() else 0.0
    ts = 2.0 if str(target_id).startswith("S2-") else 3.0

    # Context
    rel_score = coarse_score / max(top_score, 1.0) if top_score > 0 else 0.0

    return [
        nts, ntset, nr, npr, ncr, njac, ncom, ndiff, nfm, nsm,
        a_null, ats, atset, ar, apr, ajac, acom,
        nex, njn, float(len(s1n)), float(len(s2n)), pm,
        cm, ts,
        coarse_score, float(cand_rank), rel_score, float(total_cands),
    ]


# ─── LightGBM model wrapper ───

import lightgbm as lgb


class LocalModel:
    """Simplified LightGBM model for local inference."""
    def __init__(self, **params):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_threads": -1,
            "num_leaves": 41,
            "learning_rate": 0.08,
            "min_child_samples": 25,
            **params
        }
        self.model = None

    def fit(self, X, y, n_rounds=200):
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        self.params["scale_pos_weight"] = neg / max(pos, 1)
        dtrain = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, dtrain, num_boost_round=n_rounds)

    def predict_proba(self, X):
        return self.model.predict(X)


# ─── Decision rules ───

def apply_decision(s1_ids, t_ids, probs, threshold=0.68, enforce_1to1=True):
    """Greedy 1-to-1 resolution."""
    triples = [(p, s1, tid) for s1, tid, p in zip(s1_ids, t_ids, probs) if p >= threshold]
    triples.sort(reverse=True)

    accepted = defaultdict(set)
    claimed = set()
    counts = defaultdict(int)
    for prob, s1, tid in triples:
        if enforce_1to1 and tid in claimed:
            continue
        if counts[s1] >= 11:
            continue
        accepted[s1].add(tid)
        claimed.add(tid)
        counts[s1] += 1
    return dict(accepted)


# ─── Main pipeline ───

def main():
    os.makedirs("output", exist_ok=True)
    t_total = time.time()

    # 1. Train model on a sample
    print("=" * 60)
    print("STEP 1: Training production model on training sample...")
    print("=" * 60)
    t0 = time.time()

    TRAIN_SAMPLE = 15000
    s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=TRAIN_SAMPLE)
    s1_ids_set = set(s1_df["entity_id"].to_list())

    gt_df = pl.scan_csv("dataset/train/train_ground_truth.tsv", separator="\t").filter(
        pl.col("source1_entity_id").is_in(list(s1_ids_set))
    ).collect()

    gt_map = {}
    true_target_ids = set()
    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        t_str = r["matched_entity_ids"]
        if t_str and str(t_str).strip() and str(t_str).lower() != "nan":
            targets = set(str(t_str).split(","))
            gt_map[sid] = targets
            true_target_ids.update(targets)
        else:
            gt_map[sid] = set()

    # Load targets: true matches + background
    t_s2 = pl.scan_csv("dataset/train/train_source2.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(list(true_target_ids))
    ).collect()
    t_s3 = pl.scan_csv("dataset/train/train_source3.tsv", separator="\t").filter(
        pl.col("entity_id").is_in(list(true_target_ids))
    ).collect()
    bg_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=25000)
    bg_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=25000)
    targets_df = pl.concat([t_s2, t_s3, bg_s2, bg_s3]).unique(subset=["entity_id"])

    # Build training data via blocking
    print(f"Building blocking index over {len(targets_df):,} targets...")
    index = SimpleBlockingIndex()
    target_clean = {}
    for row in targets_df.iter_rows(named=True):
        tid = row["entity_id"]
        cn = normalize_name(row["business_name"])
        compact = cn.replace(" ", "")
        raw_a = row["business_address"]
        ca = normalize_address(raw_a) if raw_a and str(raw_a).lower() != "nan" else ""
        nums = [clean_num(n) for n in extract_numbers(ca)]
        postal = extract_postal_code(ca)
        country = row.get("country", "")
        target_clean[tid] = (cn, compact, ca, nums, postal, country)
        index.add_target(tid, row["business_name"], row["business_address"], str(country))

    print(f"Generating training pairs...")
    X_list, y_list = [], []
    for row in s1_df.iter_rows(named=True):
        sid = row["entity_id"]
        s1_cn = normalize_name(row["business_name"])
        s1_compact = s1_cn.replace(" ", "")
        raw_a = row["business_address"]
        s1_ca = normalize_address(raw_a) if raw_a and str(raw_a).lower() != "nan" else ""
        s1_nums = [clean_num(n) for n in extract_numbers(s1_ca)]
        s1_postal = extract_postal_code(s1_ca)
        s1_country = row.get("country", "")

        cands = index.get_candidates(sid, row["business_name"], row["business_address"],
                                     str(s1_country), max_cands=30, min_floor=24.0)
        if not cands:
            continue

        top_score = cands[0][1]
        total = len(cands)
        true_set = gt_map.get(sid, set())

        for rank, (cid, cscore) in enumerate(cands, 1):
            if cid not in target_clean:
                continue
            td = target_clean[cid]
            feat = compute_pairwise_features(
                s1_cn, s1_compact, s1_ca, s1_nums, s1_postal, s1_country,
                cid, td[0], td[1], td[2], td[3], td[4], td[5],
                cscore, rank, top_score, total
            )
            X_list.append(feat)
            y_list.append(1 if cid in true_set else 0)

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.int32)
    print(f"Training data: {X_train.shape[0]:,} pairs, {y_train.sum():,} positives ({100*y_train.mean():.2f}%)")

    model = LocalModel(num_leaves=41, learning_rate=0.08, min_child_samples=25)
    model.fit(X_train, y_train, n_rounds=200)

    # Hard negative mining round
    probs = model.predict_proba(X_train)
    weights = np.ones(len(y_train), dtype=np.float32)
    fp_mask = (probs > 0.3) & (y_train == 0)
    fn_mask = (probs < 0.7) & (y_train == 1)
    weights[fp_mask] = 2.5
    weights[fn_mask] = 2.5
    dtrain = lgb.Dataset(X_train, label=y_train, weight=weights)
    model.model = lgb.train(model.params, dtrain, num_boost_round=200)

    print(f"Model trained in {time.time() - t0:.1f}s")
    del s1_df, gt_df, targets_df, X_list, y_list
    import gc; gc.collect()

    # 2. Process test set by country
    print("\n" + "=" * 60)
    print("STEP 2: Running test inference by country...")
    print("=" * 60)

    countries = ["France", "US", "India"]
    all_matching = {}
    all_candidates = {}

    for country in countries:
        print(f"\n--- {country} ---")
        t0 = time.time()

        s1 = pl.scan_csv("dataset/test/test_source1.tsv", separator="\t").filter(
            pl.col("country") == country
        ).collect()
        s2 = pl.scan_csv("dataset/test/test_source2.tsv", separator="\t").filter(
            pl.col("country") == country
        ).collect()
        s3 = pl.scan_csv("dataset/test/test_source3.tsv", separator="\t").filter(
            pl.col("country") == country
        ).collect()
        print(f"S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")

        # Build blocking index for this country
        targets = pl.concat([s2, s3])
        idx = SimpleBlockingIndex()
        t_clean = {}
        for row in targets.iter_rows(named=True):
            tid = row["entity_id"]
            cn = normalize_name(row["business_name"])
            compact = cn.replace(" ", "")
            raw_a = row["business_address"]
            ca = normalize_address(raw_a) if raw_a and str(raw_a).lower() != "nan" else ""
            nums = [clean_num(n) for n in extract_numbers(ca)]
            postal = extract_postal_code(ca)
            t_clean[tid] = (cn, compact, ca, nums, postal, country)
            idx.add_target(tid, row["business_name"], row["business_address"], country)

        print(f"Index built for {len(targets):,} targets in {time.time() - t0:.1f}s")

        # Score all S1 entities
        all_pairs_s1, all_pairs_t, all_feats = [], [], []
        s1_rows = s1.to_dicts()

        for i, s1_r in enumerate(s1_rows):
            sid = s1_r["entity_id"]
            s1_cn = normalize_name(s1_r["business_name"])
            s1_compact = s1_cn.replace(" ", "")
            raw_a = s1_r["business_address"]
            s1_ca = normalize_address(raw_a) if raw_a and str(raw_a).lower() != "nan" else ""
            s1_nums = [clean_num(n) for n in extract_numbers(s1_ca)]
            s1_postal = extract_postal_code(s1_ca)

            cands = idx.get_candidates(sid, s1_r["business_name"], s1_r["business_address"],
                                        country, max_cands=30, min_floor=24.0)
            all_candidates[sid] = [c[0] for c in cands]

            if not cands:
                continue

            top_score = cands[0][1]
            total = len(cands)

            for rank, (cid, cscore) in enumerate(cands, 1):
                if cid not in t_clean:
                    continue
                td = t_clean[cid]
                feat = compute_pairwise_features(
                    s1_cn, s1_compact, s1_ca, s1_nums, s1_postal, country,
                    cid, td[0], td[1], td[2], td[3], td[4], td[5],
                    cscore, rank, top_score, total
                )
                all_pairs_s1.append(sid)
                all_pairs_t.append(cid)
                all_feats.append(feat)

            if (i + 1) % 50000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{country}] {i+1:,}/{len(s1_rows):,} ({rate:,.0f}/s)")

        print(f"  [{country}] Total pairs: {len(all_feats):,}")

        if all_feats:
            X_test = np.array(all_feats, dtype=np.float32)
            probs = model.predict_proba(X_test)
            matches = apply_decision(all_pairs_s1, all_pairs_t, probs,
                                     threshold=0.68, enforce_1to1=True)
            for s1_r in s1_rows:
                sid = s1_r["entity_id"]
                if sid in matches:
                    all_matching[sid] = sorted(list(matches[sid]))
                else:
                    all_matching[sid] = []
        else:
            for s1_r in s1_rows:
                all_matching[s1_r["entity_id"]] = []

        elapsed = time.time() - t0
        print(f"  [{country}] Done in {elapsed:.1f}s")
        del s1, s2, s3, targets, idx, t_clean
        gc.collect()

    # 3. Write output files
    print("\n" + "=" * 60)
    print("STEP 3: Writing output files...")
    print("=" * 60)

    s1_all = pl.scan_csv("dataset/test/test_source1.tsv", separator="\t").select("entity_id").collect()
    s1_all_ids = s1_all["entity_id"].to_list()
    print(f"Total test S1 entities: {len(s1_all_ids):,}")

    with open("output/matching_results.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            matches = all_matching.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")

    with open("output/candidate_pairs.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_all_ids:
            cands = all_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(cands)}\n")

    n_matched = sum(1 for v in all_matching.values() if v)
    n_total_matches = sum(len(v) for v in all_matching.values())
    elapsed_total = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"✅ INFERENCE COMPLETE")
    print(f"  Total time:    {elapsed_total/60:.1f} minutes")
    print(f"  S1 entities:   {len(s1_all_ids):,}")
    print(f"  Matched:       {n_matched:,}")
    print(f"  Singletons:    {len(s1_all_ids) - n_matched:,}")
    print(f"  Total matches: {n_total_matches:,}")
    print(f"  Output files:")
    print(f"    output/matching_results.tsv ({os.path.getsize('output/matching_results.tsv')/(1024*1024):.1f} MB)")
    print(f"    output/candidate_pairs.tsv ({os.path.getsize('output/candidate_pairs.tsv')/(1024*1024):.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
