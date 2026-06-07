"""
Upload severity training script to Google Cloud Storage.
Usage: python scripts/upload_to_gcs_severity.py
"""
import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "ai-grievance-sandi"
REGION       = "us-central1"
BUCKET_NAME  = f"{PROJECT_ID}-data"
BUCKET_URI   = f"gs://{BUCKET_NAME}"

GCS_DATA_DIR = f"{BUCKET_URI}/bert_training/severity"

def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
    return result

def main():
    # ── Checking bucket ───────────────────────────────────────────────────────
    print(f"\n[1/2] Checking bucket {BUCKET_URI} ...")
    check = run(f'gcloud storage buckets describe {BUCKET_URI} --format="value(name)" 2>nul', check=False)
    if check.returncode != 0:
        print(f"  Bucket not found. Creating in {REGION}...")
        run(f'gcloud storage buckets create {BUCKET_URI} --project={PROJECT_ID} '
            f'--location={REGION} --uniform-bucket-level-access')
        print(f"  [OK] Bucket created: {BUCKET_URI}")
    else:
        print(f"  [OK] Bucket exists: {BUCKET_URI}")

    # ── Upload training script ────────────────────────────────────────────────
    train_script = Path(__file__).resolve().parent / "distilbert_train_severity.py"
    if train_script.exists():
        gcs_script = f"{GCS_DATA_DIR}/distilbert_train_severity.py"
        print(f"\n[2/2] Uploading training script to {gcs_script} ...")
        run(f'gcloud storage cp "{train_script}" {gcs_script}')
        print(f"  [OK] Training script uploaded.")
    else:
        print(f"\n[2/2] Training script not found at {train_script} — skipping upload.")
        sys.exit(1)

    # ── Verify ────────────────────────────────────────────────────────────────
    print(f"\n[DONE] Listing {GCS_DATA_DIR}/:")
    run(f"gcloud storage ls -l {GCS_DATA_DIR}/")

if __name__ == "__main__":
    main()
