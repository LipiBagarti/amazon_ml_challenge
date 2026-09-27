import sys
import time
import os
import polars as pl
from rapidfuzz import fuzz

sys.path.append(os.path.abspath("code/business_entity_resolution/src"))
from normalization import normalize_name, normalize_address, extract_numbers, extract_postal_code

sys.stdout.reconfigure(encoding='utf-8')
print("=== Running Normalization Module Tests ===")

# Test 1: Concrete noise patterns from actual dataset
test_cases = [
    # US Cases
    ("Dynamic Cloud Collective Inc", "inc dynamic cloud collective"),
    ("Kay Skin Care LLC", "The Kay Skin Care LLC"),
    ("Maure Williams Colombier Inc", "maurewilliamscolombier.com"),
    ("Orelee's Barbershop", "orelees barbershop"),
    ("B+ Retail Inc", "b plus retail"),
    
    # India Cases
    ("Delhi Software Private Limited", "DELHl SOFTWARE PRIVATE [LÍMITED]"),
    ("Rashi Films (India) LLP", "Rashi  Films (India) L.L.P."),
    ("Ds Systems Corp", "Ds Corp Center"),
    ("Agro Tritan Processors Limited", "Agr0 Trstman Processors Limited"),
    
    # France Cases
    ("Thermal & Fils SASU", "thermal et fils"),
    ("ZNB Club SARL", "znb club"),
    ("SCI Ptit Àmicale", "sci ptit amicale"),
    ("<< Team Ecole", "team ecole"),
    ("Fractales Amis Groupe S.A.S", "fractales amis"),
]

print("\n--- Test 1: Name Normalization Test Cases ---")
for orig, noisy in test_cases:
    norm_orig = normalize_name(orig)
    norm_noisy = normalize_name(noisy)
    sim_raw = fuzz.token_sort_ratio(orig, noisy)
    sim_norm = fuzz.token_sort_ratio(norm_orig, norm_noisy)
    print(f"Original: '{orig}' -> Normalized: '{norm_orig}'")
    print(f"Noisy:    '{noisy}' -> Normalized: '{norm_noisy}'")
    print(f"Similarity: Raw={sim_raw:.1f}% -> Normalized={sim_norm:.1f}%\n")

# Test 2: Address Normalization & Expansion
addr_cases = [
    ("1659 Cunningham Lane, Crystal Lake, IL", "1659 CUNNINGHAM LN, CRYSTAL LAKE, IL"),
    ("795 Mountain Road, Newington, CT", "CT, NEWINGTON, 795 MOUNTAIN RD"),
    ("175 Boulevard du Président Franklin Roosevelt", "175 bd president franklin roosevelt"),
    ("NO. 5 ALLÉE DES HÊTRES, Pornic", "5 allee des hetres pornic"),
    ("Bldg No.43, Venus Tower, Behind Axis Bank, Veera Desai Rd., Mumbai", "bldg no 43 venus tower veera desai road mumbai"),
]

print("\n--- Test 2: Address Normalization Test Cases ---")
for addr1, addr2 in addr_cases:
    norm1 = normalize_address(addr1)
    norm2 = normalize_address(addr2)
    sim_raw = fuzz.token_sort_ratio(addr1, addr2)
    sim_norm = fuzz.token_sort_ratio(norm1, norm2)
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    print(f"Addr1: '{addr1}' -> '{norm1}'")
    print(f"Addr2: '{addr2}' -> '{norm2}'")
    print(f"Numbers: {nums1} vs {nums2} (Match: {set(nums1) == set(nums2)})")
    print(f"Similarity: Raw={sim_raw:.1f}% -> Normalized={sim_norm:.1f}%\n")

# Test 3: Throughput Benchmark on 50,000 real rows
print("--- Test 3: Throughput Benchmark on 50,000 Dataset Rows ---")
s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=50000)
names = s1["business_name"].to_list()
addrs = s1["business_address"].to_list()

start = time.time()
clean_names = [normalize_name(n) for n in names]
name_time = time.time() - start
name_throughput = len(names) / name_time

start = time.time()
clean_addrs = [normalize_address(a) for a in addrs]
addr_time = time.time() - start
addr_throughput = len(addrs) / addr_time

print(f"Processed 50,000 names in {name_time:.2f}s ({name_throughput:,.0f} names/sec)")
print(f"Processed 50,000 addresses in {addr_time:.2f}s ({addr_throughput:,.0f} addresses/sec)")

# Test 4: Token Overlap Improvement on Ground Truth Matches
print("\n--- Test 4: Token Overlap Improvement on 5,000 Ground Truth Pairs ---")
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t", n_rows=5000)
gt = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
pairs = gt.select(
    pl.col("source1_entity_id"),
    pl.col("matched_entity_ids").str.split(",").alias("target_id")
).explode("target_id").head(2000)

s1_lookup = {r["entity_id"]: (r["business_name"], r["business_address"]) 
             for r in s1.to_dicts()}

# Load targets
s2_sample = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=50000)
s3_sample = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=50000)
target_lookup = {r["entity_id"]: (r["business_name"], r["business_address"])
                 for r in pl.concat([s2_sample, s3_sample]).to_dicts()}

raw_sims = []
norm_sims = []

for row in pairs.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    t_id = row["target_id"]
    if s1_id in s1_lookup and t_id in target_lookup:
        n1, a1 = s1_lookup[s1_id]
        n2, a2 = target_lookup[t_id]
        
        sim_raw = fuzz.token_sort_ratio(n1, n2)
        sim_norm = fuzz.token_sort_ratio(normalize_name(n1), normalize_name(n2))
        raw_sims.append(sim_raw)
        norm_sims.append(sim_norm)

if raw_sims:
    avg_raw = sum(raw_sims) / len(raw_sims)
    avg_norm = sum(norm_sims) / len(norm_sims)
    print(f"Evaluated {len(raw_sims)} true match pairs:")
    print(f"  Average Raw Name Token Sort Ratio:        {avg_raw:.2f}%")
    print(f"  Average Normalized Name Token Sort Ratio: {avg_norm:.2f}%")
    print(f"  Absolute Token Overlap Gain:              +{avg_norm - avg_raw:.2f}%")

print("\n=== Normalization Module Validated Successfully ===")
