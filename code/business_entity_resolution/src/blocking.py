"""
Enhanced Hybrid Blocking and Candidate Generation Module.
Features:
1. Strict country partitioning (100% precision isolation)
2. Normalized name token + phonetic Soundex index
3. Compact-name index (handles concatenated tokens like 'moorebitwise.com' vs 'Moore Bitwise Inc')
4. Address numerical anchor index (leading-zero normalized building numbers + locality tokens)
5. Robust scoring floor and dynamic candidate ranking
"""

import os
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
from rapidfuzz import fuzz

sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from normalization import (
    normalize_name, normalize_address, extract_numbers, extract_postal_code, get_token_set
)

def compute_soundex(token: str) -> str:
    """Compute American Soundex code for a word token."""
    if not token:
        return ""
    token = token.upper()
    first_letter = token[0]
    
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    digits = []
    prev = mapping.get(first_letter, '0')
    for char in token[1:]:
        d = mapping.get(char, '0')
        if d != '0' and d != prev:
            digits.append(d)
        prev = d
        if len(digits) == 3:
            break
            
    digits = "".join(digits).ljust(3, '0')[:3]
    return first_letter + digits


def clean_num_token(n: str) -> str:
    """Normalize numerical tokens by stripping leading zeros (e.g. '0017560' -> '17560')."""
    stripped = n.lstrip('0')
    return stripped if stripped else "0"


class BlockingIndex:
    """
    Inverted index for rapid candidate retrieval across millions of target records.
    Partitioned by country to ensure 100% precision isolation and zero cross-country noise.
    """
    def __init__(self):
        # country -> name_token -> list of target entity_ids
        self.name_token_index = defaultdict(lambda: defaultdict(list))
        # country -> compact_name_prefix (6 chars) -> list of target entity_ids
        self.compact_name_index = defaultdict(lambda: defaultdict(list))
        # country -> soundex_key -> list of target entity_ids
        self.soundex_index = defaultdict(lambda: defaultdict(list))
        # country -> (clean_number, addr_token) -> list of target entity_ids
        self.addr_anchor_index = defaultdict(lambda: defaultdict(list))
        # country -> rare_number -> list of target entity_ids (for rare long numbers)
        self.rare_number_index = defaultdict(lambda: defaultdict(list))
        # country -> rare_addr_word -> list of target entity_ids (distinct address tokens like 'dantuluri', 'westchester')
        self.rare_addr_word_index = defaultdict(lambda: defaultdict(list))
        
        # Fast metadata lookups
        self.target_data = {}  # entity_id -> (clean_name, compact_name, clean_addr, clean_nums, country)
        
        # Stop words to prevent posting list explosions
        self.stop_words = {
            "and", "the", "in", "of", "for", "on", "at", "to", "a", "an",
            "store", "shop", "service", "services", "company", "center", "centre",
            "street", "road", "avenue", "boulevard", "lane", "drive", "court",
            "building", "floor", "apartment", "suite", "number", "near", "opposite",
            "north", "south", "east", "west", "india", "state", "city"
        }

    def add_target(self, entity_id: str, name: str, addr: str, country: str):
        """Index a single target record (S2 or S3)."""
        clean_name = normalize_name(name)
        compact_name = clean_name.replace(" ", "")
        clean_addr = normalize_address(addr) if addr and str(addr).lower() != "nan" else ""
        
        raw_nums = extract_numbers(clean_addr)
        clean_nums = [clean_num_token(n) for n in raw_nums]
        
        self.target_data[entity_id] = (clean_name, compact_name, clean_addr, clean_nums, country)
        
        c = country.strip() if country else "UNKNOWN"
        name_tokens = clean_name.split()
        
        # 1. Index significant name tokens (length >= 3, not stopword)
        indexed_tokens = set()
        for tok in name_tokens:
            if len(tok) >= 3 and tok not in self.stop_words:
                if tok not in indexed_tokens:
                    self.name_token_index[c][tok].append(entity_id)
                    indexed_tokens.add(tok)
                    
        # 2. Compact name prefix (first 6+ chars, catches 'moorebitwise' vs 'moore bitwise')
        if len(compact_name) >= 5:
            prefix = compact_name[:6]
            self.compact_name_index[c][prefix].append(entity_id)
                    
        # 3. Index Soundex of first significant name token
        for tok in name_tokens:
            if len(tok) >= 3 and tok.isalpha():
                sx = compute_soundex(tok)
                if sx:
                    self.soundex_index[c][sx].append(entity_id)
                break
                
        # 4. Multi-token Address numerical anchors and Rare Address Words
        if clean_addr:
            addr_words = [w for w in clean_addr.split() if len(w) >= 5 and not w.isdigit() and w not in self.stop_words]
            for num in clean_nums[:2]:
                if len(num) >= 4:
                    self.rare_number_index[c][num].append(entity_id)
                for atok in addr_words[:3]:
                    self.addr_anchor_index[c][(num, atok)].append(entity_id)
            # Index distinct rare address words directly
            for aw in addr_words:
                self.rare_addr_word_index[c][aw].append(entity_id)

    def get_candidates(
        self,
        s1_id: str,
        name: str,
        addr: str,
        country: str,
        max_candidates: int = 35,
        min_similarity_floor: float = 22.0
    ) -> List[Tuple[str, float]]:
        """
        Retrieve and rank candidate target IDs for a Source-1 entity.
        Returns list of (target_id, coarse_similarity_score).
        """
        clean_name = normalize_name(name)
        compact_name = clean_name.replace(" ", "")
        clean_addr = normalize_address(addr) if addr and str(addr).lower() != "nan" else ""
        raw_nums = extract_numbers(clean_addr)
        clean_nums = [clean_num_token(n) for n in raw_nums]
        c = country.strip() if country else "UNKNOWN"
        
        candidate_ids = set()
        name_tokens = clean_name.split()
        
        # Channel 1: Significant name tokens
        for tok in name_tokens:
            if len(tok) >= 3 and tok not in self.stop_words:
                postings = self.name_token_index[c].get(tok, [])
                if len(postings) <= 600:
                    candidate_ids.update(postings)
                else:
                    candidate_ids.update(postings[:600])
                    
        # Channel 2: Compact name prefix
        if len(compact_name) >= 5:
            prefix = compact_name[:6]
            postings = self.compact_name_index[c].get(prefix, [])
            candidate_ids.update(postings[:400])
                    
        # Channel 3: Soundex on first name token
        for tok in name_tokens:
            if len(tok) >= 3 and tok.isalpha():
                sx = compute_soundex(tok)
                if sx:
                    postings = self.soundex_index[c].get(sx, [])
                    if len(postings) <= 400:
                        candidate_ids.update(postings)
                    else:
                        candidate_ids.update(postings[:400])
                break
                
        # Channel 4: Address anchors & rare numbers & rare address words
        if clean_addr:
            addr_words = [w for w in clean_addr.split() if len(w) >= 5 and not w.isdigit() and w not in self.stop_words]
            for num in clean_nums[:2]:
                if len(num) >= 4:
                    postings = self.rare_number_index[c].get(num, [])
                    candidate_ids.update(postings[:300])
                for atok in addr_words[:3]:
                    postings = self.addr_anchor_index[c].get((num, atok), [])
                    candidate_ids.update(postings[:300])
            for aw in addr_words:
                postings = self.rare_addr_word_index[c].get(aw, [])
                if len(postings) <= 150:  # Rare word filter
                    candidate_ids.update(postings)
                    
        if not candidate_ids:
            return []
            
        # Score and prune candidates
        scored_candidates = []
        for tid in candidate_ids:
            t_name, t_compact, t_addr, t_nums, t_country = self.target_data[tid]
            
            # Name similarities
            sim_token_sort = fuzz.token_sort_ratio(clean_name, t_name)
            sim_compact = fuzz.ratio(compact_name, t_compact) if compact_name and t_compact else 0.0
            best_name_sim = max(sim_token_sort, sim_compact)
            
            # Address similarities
            sim_addr = 0.0
            if clean_addr and t_addr:
                sim_addr = fuzz.token_sort_ratio(clean_addr, t_addr)
                
            # Number agreement bonus
            num_overlap = bool(set(clean_nums) & set(t_nums)) if (clean_nums and t_nums) else False
            
            # Composite coarse similarity
            if num_overlap:
                coarse_score = max(best_name_sim + 15.0, sim_addr + 25.0)
            else:
                coarse_score = max(best_name_sim, sim_addr * 0.85)
            
            if coarse_score >= min_similarity_floor:
                scored_candidates.append((tid, coarse_score))
                
        # Sort descending by coarse score and take top K
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:max_candidates]
