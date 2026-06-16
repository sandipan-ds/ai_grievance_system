"""
Download DeBERTa v3 severity training results (v2) from GCS, fetch logs,
compute confusion matrix & classification metrics, and download model weights.

Usage:
  python scripts/download_results_severity_deberta_v3.py [--skip-weights]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

PROJECT_ID = "ai-grievance-sandi"
BUCKET_NAME = f"{PROJECT_ID}-data"
GCS_OUTPUT = f"gs://{BUCKET_NAME}/bert_training/severity_v2/output_deberta_v3_classifier_v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "deberta_v3_classifier"


def run(cmd: str, capture=False):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0 and not capture:
        print(f"  [ERROR] Command failed with return code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr.strip()}")
    return result


def get_latest_job_id():
    print("Finding latest DeBERTa v3 severity training job...")
    cmd = f'gcloud ai custom-jobs list --project={PROJECT_ID} --region=us-central1 --limit=15 --format="json"'
    result = run(cmd, capture=True)
    if result.returncode != 0:
        return None
    try:
        jobs = json.loads(result.stdout)
        deberta_jobs = [j for j in jobs if j.get("displayName", "").startswith("deberta-v3-severity-classifier-v2-")]
        if not deberta_jobs:
            return None
        deberta_jobs.sort(key=lambda x: x.get("createTime", ""), reverse=True)
        name = deberta_jobs[0].get("name", "")
        return name.split("/")[-1]
    except Exception as e:
        print(f"  Warning: Error parsing job list: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-weights", action="store_true", help="Skip downloading the large model weights.")
    args = parser.parse_args()

    # Create local directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify GCS output is available
    print(f"\n[1/5] Verifying GCS output path: {GCS_OUTPUT}/")
    result = run(f"gcloud storage ls {GCS_OUTPUT}/", capture=True)
    if result.returncode != 0:
        print(f"ERROR: Could not list GCS output. Is the training job done?")
        sys.exit(1)
    
    # 2. Download metrics JSON
    print(f"\n[2/5] Downloading results JSON to {MODEL_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/results_deberta_v3_classifier_v2.json" "{MODEL_DIR}/"')
    
    # 3. Download OOF predictions
    print(f"\n[3/5] Downloading OOF predictions to {MODEL_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/oof_predictions_deberta_v3_classifier_v2.json" "{MODEL_DIR}/"')

    # 4. Fetch Vertex AI Training Logs
    print(f"\n[4/5] Fetching Vertex AI training logs...")
    job_id = get_latest_job_id()
    if not job_id:
        print("  Warning: Could not automatically detect latest job ID.")
    else:
        log_file = MODEL_DIR / "training.log"
        print(f"  Latest Job ID: {job_id}")
        print(f"  Saving training logs to {log_file}...")
        import shutil
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

    # 5. Download Model Weights (Optional)
    if not args.skip_weights:
        print(f"\n[5/5] Downloading model weights for all folds to {MODEL_DIR}/")
        for fold in range(5):
            print(f"  Downloading Fold {fold} weights...")
            fold_dir = MODEL_DIR / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            # copy model weights and tokenizer
            run(f'gcloud storage cp -r "{GCS_OUTPUT}/fold_{fold}/*" "{fold_dir}/"')
        print("  [OK] Model weights saved.")
    else:
        print(f"\n[5/5] Skipping model weights download as requested.")

    # Print summary
    metrics_file = MODEL_DIR / "results_deberta_v3_classifier_v2.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"\n{'='*60}")
        print(f"  DEBERTA V3 SEVERITY CLASSIFIER — RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Avg Val Accuracy:      {m['avg_val_acc']:.4f}")
        print(f"  Avg Val F1-macro:      {m['avg_val_f1']:.4f}")
        print(f"  Avg Val F1-micro:      {m['avg_val_f1_micro']:.4f}")
        print(f"  Avg Val F1-weighted:   {m['avg_val_f1_weighted']:.4f}")
        print(f"  Avg Train F1-macro:    {m['avg_train_f1']:.4f}")
        print(f"  Avg Val Loss:          {m['avg_val_loss']:.4f}")

    print(f"\nFiles saved locally to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
