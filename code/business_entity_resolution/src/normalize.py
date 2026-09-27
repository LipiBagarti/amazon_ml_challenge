"""
Text Normalization Module for Business Entity Resolution.
Country-agnostic cleaning: accent stripping, legal suffix removal,
abbreviation expansion, numerical anchor extraction.
"""

import re
import unicodedata
from typing import Optional, Set, List


def strip_accents(text: str) -> str:
    """Strip accents/diacritical marks via NFKD decomposition."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# --- Compiled regex patterns ---
_SYMBOL_REPLACEMENTS = [
    (re.compile(r"&"), " and "),
    (re.compile(r"@"), " at "),
    (re.compile(r"\+"), " plus "),
]
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")

# Legal suffixes (multi-word first, then single-word), US / India / France / Intl.
_LEGAL_SUFFIXES = [
    r"private limited", r"pvt limited", r"pvt ltd", r"private ltd",
    r"pvt\s+ltd", r"p\s+ltd", r"pty\s+ltd",
    r"public limited company", r"public limited",
    r"limited liability company", r"limited liability partnership",
    r"societe anonyme", r"societe a responsabilite limitee",
    r"et fils", r"et freres", r"and sons", r"and brothers",
    r"and co", r"and company",
    r"incorporated", r"incorporation", r"inc", r"corporation", r"corp",
    r"limited", r"ltd", r"llc", r"l\.l\.c", r"llp", r"l\.l\.p",
    r"pllc", r"pc", r"pa",
    r"company", r"co", r"holdings", r"holding", r"group",
    r"enterprises", r"enterprise", r"international", r"intl",
    r"pvt", r"private",
    r"sarl", r"s\.a\.r\.l", r"sasu", r"s\.a\.s\.u",
    r"sas", r"s\.a\.s", r"sa", r"s\.a", r"sci", r"s\.c\.i",
    r"eurl", r"e\.u\.r\.l", r"snc", r"cie", r"groupe",
    r"fils", r"freres",
]
_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(_LEGAL_SUFFIXES) + r")\b", flags=re.IGNORECASE
)

# Address abbreviation expansions
_ADDR_MAP = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bvd": "boulevard",
    "bd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court",
    "pl": "place", "sq": "square", "pkwy": "parkway", "hwy": "highway",
    "cir": "circle", "ter": "terrace", "trl": "trail",
    "bldg": "building", "fl": "floor", "flr": "floor",
    "apt": "apartment", "ste": "suite", "rm": "room", "no": "number",
    "sec": "sector", "extn": "extension", "ext": "extension",
    "soc": "society", "col": "colony", "dist": "district",
    "opp": "opposite", "nr": "near",
    "r": "rue", "imp": "impasse", "che": "chemin", "rte": "route",
}
_ADDR_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ADDR_MAP) + r")\b",
    flags=re.IGNORECASE,
)

_NUM_RE = re.compile(r"\b\d+[a-z]?\b", flags=re.IGNORECASE)
_POSTAL_RE = re.compile(r"\b\d{5,6}\b")


# ── public API ──────────────────────────────────────────────────────

def normalize_text(text: Optional[str]) -> str:
    """General normalisation: NFKD → lowercase → symbols → punctuation."""
    if text is None:
        return ""
    text = str(text)
    if not text or text.lower() == "nan":
        return ""
    text = strip_accents(text).lower()
    for pat, repl in _SYMBOL_REPLACEMENTS:
        text = pat.sub(repl, text)
    text = _PUNCT_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_name(name: Optional[str]) -> str:
    """Normalise a business name: general clean + strip domain extensions + legal suffixes."""
    clean = normalize_text(name)
    if not clean:
        return ""
    clean = re.sub(r"\b(com|org|net|in|fr|co|gov|edu|io)\b", " ", clean)
    clean = _SUFFIX_RE.sub(" ", clean)
    clean = _MULTI_SPACE_RE.sub(" ", clean).strip()
    return clean or normalize_text(name)


def normalize_address(address: Optional[str]) -> str:
    """Normalise an address: general clean + abbreviation expansion."""
    clean = normalize_text(address)
    if not clean:
        return ""
    clean = _ADDR_RE.sub(lambda m: _ADDR_MAP[m.group(0).lower()], clean)
    return _MULTI_SPACE_RE.sub(" ", clean).strip()


def extract_numbers(text: Optional[str]) -> List[str]:
    """Extract numeric tokens (building numbers, sectors, etc.)."""
    return _NUM_RE.findall(str(text).lower()) if text else []


def extract_postal_code(text: Optional[str]) -> Optional[str]:
    """Extract a 5-/6-digit postal/PIN code (last occurrence)."""
    if not text:
        return None
    m = _POSTAL_RE.findall(str(text))
    return m[-1] if m else None


def make_record_text(name: str, address: str, country: str) -> str:
    """Combine the three fields into a single text for embedding models."""
    parts = []
    if name and str(name).lower() != "nan":
        parts.append(str(name).strip())
    if address and str(address).lower() != "nan":
        parts.append(str(address).strip())
    if country and str(country).lower() != "nan":
        parts.append(str(country).strip())
    return " | ".join(parts)
