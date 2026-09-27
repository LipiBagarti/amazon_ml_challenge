"""
Package Submission Script for Amazon ML Challenge 2026.
Assembles the complete submission archive:
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/src/         (all source, runnable)
├── code/business_entity_resolution/README.md    (exact reproduction steps)
├── code/business_entity_resolution/requirements.txt  (pinned versions)
└── Documentation_template.md
"""

import os
import sys
import zipfile
import subprocess

def create_submission_zip(team_name: str = "EntityResolvers"):
    zip_filename = f"{team_name}_submission.zip"
    print(f"=== Creating Submission Package: {zip_filename} ===")
    
    # 1. Run validator first
    print("\nRunning submission validator...")
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Validator Output:\n", res.stdout)
    if res.stderr:
        print("Validator Stderr:\n", res.stderr)
        
    if res.returncode != 0:
        print("ERROR: Validation failed! Please fix issues before packaging.")
        sys.exit(1)
        
    print("SUCCESS: Validation PASSED! Proceeding to assemble zip archive...")
    
    # 2. Files to include
    files_to_pack = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("Documentation_template.md", "Documentation_template.md"),
    ]
    
    # Include all files in code/business_entity_resolution/src/
    src_dir = "code/business_entity_resolution/src"
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            local_path = os.path.join(src_dir, fname)
            arcname = os.path.join("code/business_entity_resolution/src", fname).replace("\\", "/")
            files_to_pack.append((local_path, arcname))
            
    # 3. Create zip
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for local_path, arcname in files_to_pack:
            if os.path.isfile(local_path):
                sz_mb = os.path.getsize(local_path) / (1024*1024)
                print(f"Adding: {arcname} ({sz_mb:.2f} MB)")
                zipf.write(local_path, arcname)
            else:
                print(f"WARNING: File not found: {local_path}")
                
    total_sz = os.path.getsize(zip_filename) / (1024*1024)
    print(f"\nSuccessfully created {zip_filename} ({total_sz:.2f} MB)!")

if __name__ == "__main__":
    create_submission_zip()
