"""
Download DeBERTa v3 civic agency v2 training results from GCS.
Usage: python scripts/download_results_civic_deberta_v3.py
"""
import subprocess
import sys
import json
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "ai-grievance-sandi"
BUCKET_NAME  = f"{PROJECT_ID}-data"
GCS_OUTPUT   = f"gs://{BUCKET_NAME}/bert_training/civic_v2/deberta_v3_output"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR   = PROJECT_ROOT / "charts_and_graphs"
METRICS_DIR  = PROJECT_ROOT / "metrics_model_civic_bodies" / "deberta_v3"
DATA_DIR     = PROJECT_ROOT / "data" / "processed"


def run(cmd: str, capture=False):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0 and not capture:
        print(f"  STDERR: {result.stderr.strip()}")
    return result


def get_latest_job_id():
    print("Finding latest DeBERTa v3 civic training job...")
    cmd = f'gcloud ai custom-jobs list --project={PROJECT_ID} --region=us-central1 --limit=15 --format="json"'
    result = run(cmd, capture=True)
    if result.returncode != 0:
        return None
    try:
        jobs = json.loads(result.stdout)
        deberta_jobs = [j for j in jobs if j.get("displayName", "").startswith("deberta-v3-civic-v2-")]
        if not deberta_jobs:
            return None
        deberta_jobs.sort(key=lambda x: x.get("createTime", ""), reverse=True)
        name = deberta_jobs[0].get("name", "")
        return name.split("/")[-1]
    except Exception as e:
        print(f"  Warning: Error parsing job list: {e}")
        return None


def main():
    # ── Check what's available in GCS ─────────────────────────────────────────
    print(f"[1/5] Listing results in {GCS_OUTPUT}/")
    result = run(f"gcloud storage ls {GCS_OUTPUT}/", capture=True)
    if result.returncode != 0:
        print("ERROR: Could not list GCS output. Is the training job done?")
        sys.exit(1)
    print(result.stdout)

    # ── Create local directories ──────────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download metrics JSON ─────────────────────────────────────────────────
    print(f"[2/5] Downloading metrics to {METRICS_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/results_deberta_v3_civic.json" "{METRICS_DIR}/"')
    print(f"  [OK] Metrics saved.")

    # ── Download confusion matrix ─────────────────────────────────────────────
    print(f"\n[3/5] Downloading confusion matrix to {METRICS_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/confusion_matrix_deberta_v3_civic.npy" "{METRICS_DIR}/"')
    print(f"  [OK] Confusion matrix saved.")

    # ── Download OOF predictions ──────────────────────────────────────────────
    print(f"\n[4/5] Downloading OOF predictions to {DATA_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/oof_predictions_civic_deberta_v3.joblib" "{DATA_DIR}/oof_predictions_civic_deberta_v3.joblib"')
    print(f"  [OK] OOF predictions saved.")

    # ── Fetch Vertex AI Training Logs ─────────────────────────────────────────
    print(f"\n[5/5] Fetching Vertex AI training logs...")
    job_id = get_latest_job_id()
    if not job_id:
        print("  Warning: Could not automatically detect latest job ID.")
    else:
        log_file = METRICS_DIR / "training.log"
        print(f"  Latest Job ID: {job_id}")
        print(f"  Saving training logs to {log_file}...")
        gcloud_bin = shutil.which("gcloud") or "gcloud"
        log_cmd = [
            gcloud_bin, "logging", "read",
            f'resource.labels.job_id="{job_id}"',
            f"--project={PROJECT_ID}",
            "--order=asc",
            "--limit=20000",
            "--format=value(textPayload)"
        ]
        print(f"  $ {' '.join(log_cmd)}")
        log_result = subprocess.run(log_cmd, capture_output=True, text=True)
        if log_result.returncode == 0:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(log_result.stdout)
            print("  [OK] Logs saved.")
        else:
            print(f"  Warning: Failed to fetch logs: {log_result.stderr.strip()}")

    # ── Print summary ─────────────────────────────────────────────────────────
    metrics_file = METRICS_DIR / "results_deberta_v3_civic.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"\n{'='*60}")
        print(f"  DEBERTA V3 CIVIC AGENCY v2 — RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Avg Val Accuracy:            {m['avg_val_accuracy']:.4f}")
        print(f"  Avg Val F1-macro:            {m['avg_val_f1_macro']:.4f}")
        print(f"  Avg Val F1-micro:            {m['avg_val_f1_micro']:.4f}")
        print(f"  Avg Val F1-weighted:         {m['avg_val_f1_weighted']:.4f}")
        print(f"  Avg Val Precision (Macro):   {m['avg_val_precision_macro']:.4f}")
        print(f"  Avg Val Recall (Macro):      {m['avg_val_recall_macro']:.4f}")

    print(f"\nFiles saved:")
    print(f"  Metrics:    {METRICS_DIR / 'results_deberta_v3_civic.json'}")
    print(f"  CM:         {METRICS_DIR / 'confusion_matrix_deberta_v3_civic.npy'}")
    print(f"  OOF preds:  {DATA_DIR / 'oof_predictions_civic_deberta_v3.joblib'}")
    print(f"  Logs:       {METRICS_DIR / 'training.log'}")


if __name__ == "__main__":
    main()
