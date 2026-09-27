# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An end-to-end, high-performance, precision-weighted machine learning pipeline for cross-source Business Entity Resolution under strict memory (< 2.5 GB RAM) and CPU constraints.

---

## 1. System Overview & Architecture

Given noisy business records across three independent data sources (`Source 1`, `Source 2`, `Source 3`) with `business_name`, `business_address`, and `country`, this pipeline resolves all matching records in `Source 2` and `Source 3` for each `Source 1` reference entity.

The solution is specifically designed and optimized for **macro-averaged $F_{0.5}$** (precision weighted $2\times$ over recall) and accounts for singletons (entities with 0 matches scoring 1.0 for empty predictions and 0.0 for any false positive).

```
Raw Data (S1, S2, S3)
   │
   ▼
[1. Country-Agnostic Normalization Engine]
   ├─ NFKD accent & diacritic stripping (e.g. French accents: À, é, ê -> A, e, e)
   ├─ Multilingual legal suffix stripping (US/India/France: Inc, LLC, Pvt Ltd, SARL, SASU, SCI, etc.)
   ├─ Address abbreviation expansion (st -> street, rd -> road, ave -> avenue, etc.)
   └─ Leading-zero numerical anchor normalization (0017560 -> 17560)
   │
   ▼
[2. Multi-Channel Inverted Index Blocking Engine]
   ├─ Channel 1: Strict Country Partitioning (100% precision isolation: US, India, France)
   ├─ Channel 2: Normalized Name Tokens & Soundex Phonetic Keys
   ├─ Channel 3: Compact Domain & Name Prefix Matching (e.g. moorebitwise.com <-> Moore Bitwise)
   ├─ Channel 4: Multi-Token Building Number & Street Anchors
   └─ Channel 5: Rare Locality & City Inverted Index (recovers non-Latin Indian script records)
   │  --> Candidate Recall Ceiling: 98.17% (99.96% search space reduction)
   │
   ▼
[3. Pairwise Feature Engineering (28 Dimensional Matrix)]
   ├─ Name Lexical & String Metrics: RapidFuzz Token Sort, Token Set, Ratio, Partial, Jaccard
   ├─ Address Similarities & Null Indicators: Token Overlap, Common Tokens, Address Null Flag
   ├─ Numerical & Postal Anchors: Number Exact Match, Number Jaccard, Postal Code Match
   └─ Intra-Entity Contextual Signals: Intra-Entity Rank, Score Margin from Top Candidate
   │
   ▼
[4. Gradient Boosted Matching Model (LightGBM)]
   ├─ Histogram-based GBDT tuned for precision-heavy asymmetric classification
   └─ Hard Negative Mining: High-confidence false positives upweighted 2.5x to sharpen decision boundaries
   │
   ▼
[5. Precision-Heavy Decision Layer & Global 1-to-1 Resolution]
   ├─ Optimal Thresholding (tau = 0.68) maximizing macro F_0.5 directly
   ├─ Global 1-to-1 Bipartite Greedy Resolver: Guarantees each target record is claimed by at most one S1 entity
   └─ Singleton Safeguard: Low-confidence entities correctly predicted as empty lists (1.0 credit)
   │
   ▼
Final Outputs:
   ├─ output/matching_results.tsv (final entity matches)
   └─ output/candidate_pairs.tsv (blocking candidates fed to model)
```

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── requirements.txt       # Pinned dependencies
├── README.md              # Pipeline documentation & run instructions
└── src/
    ├── normalization.py   # Text pre-processing, suffix stripping, anchor extractors
    ├── blocking.py        # Multi-channel partitioned inverted index candidate generator
    ├── features.py        # 28-dimensional pairwise feature engineering
    ├── dataset_builder.py # Entity-level train/validation pair dataset construction
    ├── model.py           # LightGBM classifier with hard-negative mining
    ├── decision.py        # Precision thresholding & global 1-to-1 bipartite greedy resolver
    ├── evaluation.py      # Official competition macro F_0.5 scoring metric
    ├── cross_validation.py# 5-fold cross-validation pipeline
    └── inference.py       # Full production inference pipeline
```

---

## 3. Installation & Requirements

Python 3.8+ (tested on Python 3.10 to 3.14). Install dependencies via:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 4. End-to-End Execution Guide

### Option A: Run 5-Fold Cross-Validation

To verify the model's out-of-fold generalization, candidate recall ceiling, and macro $F_{0.5}$ score:

```bash
python code/business_entity_resolution/src/cross_validation.py
```

### Option B: Run Full Production Inference

To train the production model on ground truth and run full candidate generation and matching over the test set:

```bash
python code/business_entity_resolution/src/inference.py
```

This generates both required submission files in `output/`:
- `output/matching_results.tsv`: `source1_entity_id\tmatched_entity_ids`
- `output/candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids`

### Option C: Validate Outputs

Run the official competition validation script:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
