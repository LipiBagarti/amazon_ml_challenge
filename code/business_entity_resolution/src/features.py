"""
Pairwise Feature Engineering for candidate pairs.
All features are cheap CPU operations (RapidFuzz + regex) — no GPU needed.
"""
from typing import List, Optional
from rapidfuzz import fuzz
from normalize import normalize_name, normalize_address, extract_numbers, extract_postal_code
import re

FEATURE_NAMES = [
    # Name (10)
    "name_token_sort_ratio", "name_token_set_ratio", "name_ratio",
    "name_partial_ratio", "name_compact_ratio",
    "name_jaccard", "name_common_tokens", "name_len_diff_ratio",
    "name_first_token_match", "name_soundex_match",
    # Address (7)
    "addr_is_null", "addr_token_sort_ratio", "addr_token_set_ratio",
    "addr_ratio", "addr_partial_ratio", "addr_jaccard", "addr_common_tokens",
    # Numerical anchors (5)
    "num_exact_match", "num_jaccard", "num_count_s1", "num_count_t", "postal_match",
    # Structural (2)
    "country_match", "target_source",
    # Contextual (3)
    "blocking_score", "candidate_rank", "total_candidates",
]


def _soundex(token: str) -> str:
    if not token:
        return ""
    t = token.upper()
    first = t[0]
    m = {'B':'1','F':'1','P':'1','V':'1','C':'2','G':'2','J':'2','K':'2',
         'Q':'2','S':'2','X':'2','Z':'2','D':'3','T':'3','L':'4',
         'M':'5','N':'5','R':'6'}
    d, prev = [], m.get(first, '0')
    for c in t[1:]:
        v = m.get(c, '0')
        if v != '0' and v != prev:
            d.append(v)
        prev = v
        if len(d) == 3:
            break
    return first + "".join(d).ljust(3, '0')[:3]


def _strip_leading_zeros(n: str) -> str:
    s = n.lstrip('0')
    return s if s else "0"


def compute_features(
    s1_name: str, s1_addr: str, s1_country: str,
    t_name: str, t_addr: str, t_country: str,
    target_id: str,
    blocking_score: float = 0.0,
    candidate_rank: int = 0,
    total_candidates: int = 0,
) -> List[float]:
    """Compute 27-dimensional feature vector for one (s1, target) pair."""

    cn1 = normalize_name(s1_name)
    cn2 = normalize_name(t_name)
    ca1 = normalize_address(s1_addr)
    ca2 = normalize_address(t_addr)
    comp1 = cn1.replace(" ", "")
    comp2 = cn2.replace(" ", "")

    # Name lexical
    nts = fuzz.token_sort_ratio(cn1, cn2) / 100.0
    ntset = fuzz.token_set_ratio(cn1, cn2) / 100.0
    nr = fuzz.ratio(cn1, cn2) / 100.0
    npr = fuzz.partial_ratio(cn1, cn2) / 100.0
    ncr = fuzz.ratio(comp1, comp2) / 100.0 if comp1 and comp2 else 0.0

    tok1 = set(cn1.split())
    tok2 = set(cn2.split())
    inter = tok1 & tok2
    union = tok1 | tok2
    njac = len(inter) / len(union) if union else 0.0
    ncom = float(len(inter))
    ndiff = abs(len(cn1) - len(cn2)) / max(len(cn1), len(cn2), 1)

    ft1 = cn1.split()[0] if cn1 else ""
    ft2 = cn2.split()[0] if cn2 else ""
    nfm = 1.0 if ft1 and ft1 == ft2 else 0.0
    nsm = 1.0 if ft1 and _soundex(ft1) == _soundex(ft2) else 0.0

    # Address
    a_null = 1.0 if (not ca1 or not ca2) else 0.0
    if a_null == 0.0:
        ats = fuzz.token_sort_ratio(ca1, ca2) / 100.0
        atset = fuzz.token_set_ratio(ca1, ca2) / 100.0
        ar = fuzz.ratio(ca1, ca2) / 100.0
        apr = fuzz.partial_ratio(ca1, ca2) / 100.0
        at1 = set(ca1.split()); at2 = set(ca2.split())
        aint = at1 & at2; aun = at1 | at2
        ajac = len(aint) / len(aun) if aun else 0.0
        acom = float(len(aint))
    else:
        ats = atset = ar = apr = ajac = 0.0; acom = 0.0

    # Numerical anchors
    n1 = [_strip_leading_zeros(x) for x in extract_numbers(ca1)]
    n2 = [_strip_leading_zeros(x) for x in extract_numbers(ca2)]
    s1n = set(n1); s2n = set(n2)
    nex = 1.0 if s1n and s2n and (s1n & s2n) else 0.0
    nun = s1n | s2n
    njn = len(s1n & s2n) / len(nun) if nun else 0.0
    p1 = extract_postal_code(ca1)
    p2 = extract_postal_code(ca2)
    pm = 1.0 if (p1 and p2 and p1 == p2) else (-1.0 if (p1 and p2) else 0.0)

    # Structural
    cm = 1.0 if s1_country and t_country and s1_country.strip().lower() == t_country.strip().lower() else 0.0
    ts = 2.0 if target_id.startswith("S2-") else 3.0

    return [
        nts, ntset, nr, npr, ncr,
        njac, ncom, ndiff, nfm, nsm,
        a_null, ats, atset, ar, apr, ajac, acom,
        nex, njn, float(len(s1n)), float(len(s2n)), pm,
        cm, ts,
        blocking_score, float(candidate_rank), float(total_candidates),
    ]
