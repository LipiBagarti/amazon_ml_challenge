"""
Normalization Module for Business Entity Resolution.
Country-agnostic text pre-processing, legal suffix stripping,
abbreviation expansion, accent normalization, and token extraction.
"""

import re
import unicodedata
from typing import Optional, Set, List, Tuple

# Precompiled regex patterns for performance

# Accents / diacritics stripping
def strip_accents(text: str) -> str:
    """Strip accents and diacritical marks using NFKD decomposition."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

# Common symbol expansions
SYMBOL_REPLACEMENTS = [
    (re.compile(r"&"), " and "),
    (re.compile(r"@"), " at "),
    (re.compile(r"\+"), " plus "),
    (re.compile(r"%"), " percent "),
]

# Punctuation to whitespace (preserves alphanumeric characters and unicode letters)
PUNCT_REGEX = re.compile(r"[^\w\s]", flags=re.UNICODE)
MULTI_SPACE_REGEX = re.compile(r"\s+")

# Multilingual Legal Suffixes (US, India, France, International)
# Order: longest multi-word patterns first, then single tokens
LEGAL_SUFFIX_WORDS = [
    # Multi-word
    r"private limited", r"pvt limited", r"pvt ltd", r"private ltd",
    r"pvt\s+ltd", r"p\s+ltd", r"pty\s+ltd",
    r"public limited company", r"public limited",
    r"limited liability company", r"limited liability partnership",
    r"joint stock company", r"societe anonyme", r"societe a responsabilite limitee",
    r"et fils", r"et freres", r"and sons", r"and brothers", r"and co", r"and company",
    # Single tokens (English / International)
    r"incorporated", r"incorporation", r"inc", r"corporation", r"corp",
    r"limited", r"ltd", r"llc", r"l\.l\.c", r"llp", r"l\.l\.p", r"pllc", r"pc", r"pa",
    r"company", r"co", r"holdings", r"holding", r"group", r"services",
    r"enterprises", r"enterprise", r"international", r"intl", r"industries",
    r"solutions", r"consulting", r"associates", r"partners", r"ventures",
    r"technologies", r"systems", r"center", r"centre",
    # Single tokens (India)
    r"pvt", r"private",
    # Single tokens (France / European)
    r"sarl", r"s\.a\.r\.l", r"sasu", r"s\.a\.s\.u", r"sas", r"s\.a\.s",
    r"sa", r"s\.a", r"sci", r"s\.c\.i", r"eurl", r"e\.u\.r\.l",
    r"snc", r"cie", r"groupe", r"fils", r"freres", r"distribution",
]

# Build regex to strip legal suffixes at word boundaries (end or start of business name)
# e.g., 'Dynamic Cloud Collective Inc' or 'Inc Dynamic Cloud Collective'
_SUFFIX_PATTERN_STR = r"\b(?:" + "|".join(LEGAL_SUFFIX_WORDS) + r")\b"
SUFFIX_REGEX = re.compile(_SUFFIX_PATTERN_STR, flags=re.IGNORECASE)

# Address abbreviations expansion dictionary
ADDRESS_EXPANSIONS = {
    # Street types (US / UK / International)
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "bd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "pkwy": "parkway",
    "pky": "parkway",
    "hwy": "highway",
    "cir": "circle",
    "ter": "terrace",
    "terr": "terrace",
    "trl": "trail",
    "way": "way",
    "aly": "alley",
    # Units / Sub-address
    "bldg": "building",
    "fl": "floor",
    "flr": "floor",
    "apt": "apartment",
    "ste": "suite",
    "rm": "room",
    "no": "number",
    # India address markers
    "sec": "sector",
    "sect": "sector",
    "extn": "extension",
    "ext": "extension",
    "soc": "society",
    "col": "colony",
    "dist": "district",
    "opp": "opposite",
    "nr": "near",
    # France address markers
    "r": "rue",
    "all": "allee",
    "imp": "impasse",
    "che": "chemin",
    "rte": "route",
    "pl": "place",
    "av": "avenue",
}

# Regex to substitute whole-word address abbreviations
_ADDR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ADDRESS_EXPANSIONS.keys()) + r")\b",
    flags=re.IGNORECASE
)

def expand_address_abbreviations(text: str) -> str:
    """Expand standard street and address abbreviations to full words."""
    return _ADDR_PATTERN.sub(lambda m: ADDRESS_EXPANSIONS[m.group(0).lower()], text)

# Extract numbers (building numbers, sector numbers, zip/PIN codes)
NUMBER_REGEX = re.compile(r"\b\d+[a-z]?\b", flags=re.IGNORECASE)

# Extract 5 to 6 digit postal / PIN codes
POSTAL_CODE_REGEX = re.compile(r"\b\d{5,6}\b")


def normalize_text_general(text: Optional[str]) -> str:
    """General text normalization: NFKD, lowercase, symbol expansion, punctuation to space."""
    if text is None:
        return ""
    text = str(text)
    if text == "" or text.lower() == "nan":
        return ""
    
    # 1. NFKD accent stripping
    text = strip_accents(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Symbol expansion
    for pattern, repl in SYMBOL_REPLACEMENTS:
        text = pattern.sub(repl, text)
        
    # 4. Remove punctuation
    text = PUNCT_REGEX.sub(" ", text)
    
    # 5. Clean whitespace
    text = MULTI_SPACE_REGEX.sub(" ", text).strip()
    return text


def normalize_name(name: Optional[str]) -> str:
    """
    Normalize business entity name:
    - General normalization
    - Strip domain suffixes (e.g. '.com', '.org', '.in', '.fr')
    - Strip legal suffixes (e.g. 'inc', 'llc', 'pvt ltd', 'sarl')
    - Deduplicate multiple spaces
    """
    clean = normalize_text_general(name)
    if not clean:
        return ""
    
    # Strip web domain extensions if present (e.g. 'maurewilliamscolombier com')
    clean = re.sub(r"\b(com|org|net|in|fr|co|gov|edu|io)\b", " ", clean)
    
    # Strip legal suffixes
    clean = SUFFIX_REGEX.sub(" ", clean)
    
    # Final whitespace cleanup
    clean = MULTI_SPACE_REGEX.sub(" ", clean).strip()
    
    # If stripping suffixes cleared the entire name (e.g. record was literally 'Inc'), revert to general
    if not clean:
        return normalize_text_general(name)
    return clean


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize business address:
    - General normalization
    - Expand street / building / administrative abbreviations
    - Clean whitespace
    """
    clean = normalize_text_general(address)
    if not clean:
        return ""
    
    # Expand address abbreviations
    clean = expand_address_abbreviations(clean)
    
    # Final whitespace cleanup
    clean = MULTI_SPACE_REGEX.sub(" ", clean).strip()
    return clean


def extract_numbers(text: Optional[str]) -> List[str]:
    """Extract all alphanumeric numerical tokens (e.g., '123', '795', '21b', '104')."""
    if not text:
        return []
    return NUMBER_REGEX.findall(str(text).lower())


def extract_postal_code(text: Optional[str]) -> Optional[str]:
    """Extract a 5-digit (US/France) or 6-digit (India) postal code if present."""
    if not text:
        return None
    matches = POSTAL_CODE_REGEX.findall(str(text))
    if matches:
        return matches[-1]  # Postal codes usually appear toward the end of an address
    return None


def get_token_set(text: str) -> Set[str]:
    """Return the set of non-empty tokens in normalized text."""
    if not text:
        return set()
    return set(text.split())
