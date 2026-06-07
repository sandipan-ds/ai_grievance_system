"""
Download BERT civic agency training results from GCS.
Usage: python scripts/download_results_civic.py
"""
import subprocess
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "ai-grievance-sandi"
BUCKET_NAME  = f"{PROJECT_ID}-data"
GCS_OUTPUT   = f"gs://{BUCKET_NAME}/bert_training/civic/output"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR   = PROJECT_ROOT / "charts_and_graphs"
METRICS_DIR  = PROJECT_ROOT / "metrics_model_civic_bodies" / "bert"
DATA_DIR     = PROJECT_ROOT / "data" / "processed"


def run(cmd: str):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
    return result


def main():
    # ── Check what's available in GCS ─────────────────────────────────────────
    print(f"[1/4] Listing results in {GCS_OUTPUT}/")
    result = run(f"gcloud storage ls {GCS_OUTPUT}/")
    if result.returncode != 0:
        print("ERROR: Could not list GCS output. Is the training job done?")
        print("Check status: gcloud ai custom-jobs list "
              f"--project={PROJECT_ID} --region=us-central1")
        sys.exit(1)
    print(result.stdout)

    # ── Create local directories ──────────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download metrics JSON ─────────────────────────────────────────────────
    print(f"[2/4] Downloading metrics to {METRICS_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/results_bert_civic.json" "{METRICS_DIR}/"')
    print(f"  [OK] Metrics saved.")

    # ── Download confusion matrix ─────────────────────────────────────────────
    print(f"\n[3/4] Downloading confusion matrix to {METRICS_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/confusion_matrix_bert_civic.npy" "{METRICS_DIR}/"')
    print(f"  [OK] Confusion matrix saved.")

    # ── Download OOF predictions ──────────────────────────────────────────────
    print(f"\n[4/4] Downloading OOF predictions to {DATA_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/oof_predictions_civic.joblib" "{DATA_DIR}/"')
    print(f"  [OK] OOF predictions saved.")

    # ── Print summary ─────────────────────────────────────────────────────────
    import json
    metrics_file = METRICS_DIR / "results_bert_civic.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"\n{'='*60}")
        print(f"  BERT CIVIC AGENCY — RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Avg Val Accuracy:     {m['avg_val_accuracy']:.4f}")
        print(f"  Avg Val F1-macro:     {m['avg_val_f1_macro']:.4f}")
        print(f"  Avg Val F1-weighted:  {m['avg_val_f1_weighted']:.4f}")
        print(f"  Avg Val Precision:    {m['avg_val_precision']:.4f}")
        print(f"  Avg Val Recall:       {m['avg_val_recall']:.4f}")

    print(f"\nFiles saved:")
    print(f"  Metrics:    {METRICS_DIR / 'results_bert_civic.json'}")
    print(f"  CM:         {METRICS_DIR / 'confusion_matrix_bert_civic.npy'}")
    print(f"  OOF preds:  {DATA_DIR / 'oof_predictions_civic.joblib'}")


if __name__ == "__main__":
    main()
