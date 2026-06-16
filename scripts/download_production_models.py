"""
Download Fold 0 models for production deployment from GCS.
- DeBERTa v3 fold 0 (for Civic Agency WSV ensemble)
- T5 base reason fold 0 (for Severity reason generator)
"""
import subprocess
from pathlib import Path

PROJECT_ID = "ai-grievance-sandi"
BUCKET_NAME = f"{PROJECT_ID}-data"
BASE_DIR = Path(r"c:\Users\sandi\Desktop\ML Working Folder\ai_grievance_system")

def run(cmd: str):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Error executing command: {cmd}")
    return res.returncode == 0

def main():
    print("=== DOWNLOADING DEBERTA V3 CIVIC MODEL (FOLD 3) ===")
    gcs_deberta = f"gs://{BUCKET_NAME}/bert_training/civic_v2/deberta_v3_output/fold_3"
    local_deberta = BASE_DIR / "models" / "civic_bodies" / "dataset_v2" / "DeBERTa_v3" / "fold_3"
    local_deberta.mkdir(parents=True, exist_ok=True)
    run(f'gcloud storage cp -r "{gcs_deberta}/*" "{local_deberta}/"')

    print("\n=== DOWNLOADING T5 BASE REASON MODEL (FOLD 0) ===")
    gcs_t5 = f"gs://{BUCKET_NAME}/bert_training/severity_v2/output_t5_base_reason/fold_0"
    local_t5 = BASE_DIR / "models" / "severity" / "dataset_v2" / "t5_base_reason" / "fold_0"
    local_t5.mkdir(parents=True, exist_ok=True)
    run(f'gcloud storage cp -r "{gcs_t5}/*" "{local_t5}/"')

    print("\nModel downloads complete.")

if __name__ == "__main__":
    main()
