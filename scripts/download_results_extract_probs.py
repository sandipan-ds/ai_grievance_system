"""
Download probability OOF files from the stacking extraction job.
After downloading, run ensemble_soft_stacking.py locally.

Usage: python scripts/download_results_extract_probs.py
"""
import subprocess
import sys
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "ai-grievance-sandi"
BUCKET_NAME  = f"{PROJECT_ID}-data"
GCS_OUTPUT   = f"gs://{BUCKET_NAME}/bert_training/civic_v2/stacking_probs_output"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
ENSEMBLE_DIR = PROJECT_ROOT / "models" / "civic_bodies" / "dataset_v2" / "ensemble_stacking"


def run(cmd: str, capture=False):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and not capture:
        print(f"  STDERR: {result.stderr.strip() if result.stderr else ''}")
    return result


def main():
    # ── Check what's available ────────────────────────────────────────────────
    print(f"[1/4] Listing results in {GCS_OUTPUT}/")
    result = run(f"gcloud storage ls {GCS_OUTPUT}/", capture=True)
    if result.returncode != 0:
        print("ERROR: Could not list GCS output. Is the extraction job done?")
        sys.exit(1)
    print(result.stdout)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download probability OOF files ────────────────────────────────────────
    oof_files = [
        "oof_predictions_civic_distilbert_probs.joblib",
        "oof_predictions_civic_roberta_probs.joblib",
        "oof_predictions_civic_deberta_v3_probs.joblib",
    ]

    print(f"\n[2/4] Downloading probability OOF files to {DATA_DIR}/")
    for f in oof_files:
        run(f'gcloud storage cp "{GCS_OUTPUT}/{f}" "{DATA_DIR}/{f}"')
    print(f"  [OK] OOF files downloaded.")

    # ── Download extraction summary ───────────────────────────────────────────
    print(f"\n[3/4] Downloading extraction summary")
    run(f'gcloud storage cp "{GCS_OUTPUT}/extraction_summary.json" "{ENSEMBLE_DIR}/extraction_summary.json"')

    # ── Print summary ─────────────────────────────────────────────────────────
    summary_file = ENSEMBLE_DIR / "extraction_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"\n{'='*60}")
        print(f"  PROBABILITY EXTRACTION — RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Device: {summary.get('device', 'N/A')}")
        for m in summary.get("models", []):
            print(f"  {m['model']:15s}  Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
                  f"Latency={m['avg_latency_ms']:.2f}ms  Probs={m['probs_shape']}")

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n[4/4] Next steps:")
    print(f"  1. Update OOF_PATHS in ensemble_soft_stacking.py to point to the new files:")
    for f in oof_files:
        print(f"     {DATA_DIR / f}")
    print(f"  2. Run: python scripts/ensemble_soft_stacking.py")
    print(f"  3. Compare weighted soft voting vs probability stacking vs individual models")


if __name__ == "__main__":
    main()
