"""
Upload civic agency fold data to Google Cloud Storage.
Usage: python scripts/upload_to_gcs_civic.py
"""
import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "ai-grievance-sandi"
REGION       = "us-central1"
BUCKET_NAME  = f"{PROJECT_ID}-data"
BUCKET_URI   = f"gs://{BUCKET_NAME}"

LOCAL_DATA   = Path(__file__).resolve().parent.parent / "data" / "processed" / "cv_fold_data_final.joblib"
GCS_DATA_DIR = f"{BUCKET_URI}/bert_training/civic"
GCS_DATA_DST = f"{GCS_DATA_DIR}/cv_fold_data_final.joblib"


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
    return result


def main():
    # ── Verify local file exists ──────────────────────────────────────────────
    if not LOCAL_DATA.exists():
        print(f"ERROR: Local data file not found:\n  {LOCAL_DATA}")
        sys.exit(1)
    print(f"Local data: {LOCAL_DATA}  ({LOCAL_DATA.stat().st_size / 1024**2:.1f} MB)")

    # ── Create bucket if needed ───────────────────────────────────────────────
    print(f"\n[1/3] Checking bucket {BUCKET_URI} ...")
    check = run(f'gcloud storage buckets describe {BUCKET_URI} --format="value(name)" 2>nul', check=False)
    if check.returncode != 0:
        print(f"  Bucket not found. Creating in {REGION}...")
        run(f'gcloud storage buckets create {BUCKET_URI} --project={PROJECT_ID} '
            f'--location={REGION} --uniform-bucket-level-access')
        print(f"  [OK] Bucket created: {BUCKET_URI}")
    else:
        print(f"  [OK] Bucket exists: {BUCKET_URI}")

    # ── Upload data ───────────────────────────────────────────────────────────
    print(f"\n[2/3] Uploading data to {GCS_DATA_DST} ...")
    run(f'gcloud storage cp "{LOCAL_DATA}" {GCS_DATA_DST}')
    print(f"  [OK] Data uploaded.")

    # ── Upload training script ────────────────────────────────────────────────
    train_script = Path(__file__).resolve().parent / "bert_train_civic.py"
    if train_script.exists():
        gcs_script = f"{GCS_DATA_DIR}/bert_train_civic.py"
        print(f"\n[3/3] Uploading training script to {gcs_script} ...")
        run(f'gcloud storage cp "{train_script}" {gcs_script}')
        print(f"  [OK] Training script uploaded.")
    else:
        print(f"\n[3/3] Training script not found at {train_script} — skipping upload.")
        print(f"  Create it first, then re-run or upload manually.")

    # ── Verify ────────────────────────────────────────────────────────────────
    print(f"\n[DONE] Listing {GCS_DATA_DIR}/:")
    run(f"gcloud storage ls -l {GCS_DATA_DIR}/")


if __name__ == "__main__":
    main()
