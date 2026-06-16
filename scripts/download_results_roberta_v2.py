"""
Download RoBERTa severity training results (v2) from GCS, fetch logs,
compute confusion matrix & classification metrics, and download model weights.

Usage:
  python scripts/download_results_roberta_v2.py [--skip-weights]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

PROJECT_ID = "ai-grievance-sandi"
BUCKET_NAME = f"{PROJECT_ID}-data"
GCS_OUTPUT = f"gs://{BUCKET_NAME}/bert_training/severity_v2/output_roberta_classifier_v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "trial_5_roberta_classifier"

def run(cmd: str, capture=False):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if result.returncode != 0 and not capture:
        print(f"  [ERROR] Command failed with return code {result.returncode}")
        if result.stderr:
            print(f"  STDERR: {result.stderr.strip()}")
    return result

def get_latest_job_id():
    print("Finding latest RoBERTa severity training job...")
    cmd = f'gcloud ai custom-jobs list --project={PROJECT_ID} --region=us-central1 --limit=15 --format="json"'
    result = run(cmd, capture=True)
    if result.returncode != 0:
        return None
    try:
        jobs = json.loads(result.stdout)
        roberta_jobs = [j for j in jobs if j.get("displayName", "").startswith("roberta-severity-classifier-v2-")]
        if not roberta_jobs:
            return None
        roberta_jobs.sort(key=lambda x: x.get("createTime", ""), reverse=True)
        name = roberta_jobs[0].get("name", "")
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
    print(f"\n[1/6] Verifying GCS output path: {GCS_OUTPUT}/")
    result = run(f"gcloud storage ls {GCS_OUTPUT}/", capture=True)
    if result.returncode != 0:
        print(f"ERROR: Could not list GCS output. Is the training job done?")
        sys.exit(1)
    
    # 2. Download metrics JSON
    print(f"\n[2/6] Downloading results JSON to {MODEL_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/results_roberta_classifier_v2.json" "{MODEL_DIR}/"')
    
    # 3. Download OOF predictions
    print(f"\n[3/6] Downloading OOF predictions to {MODEL_DIR}/")
    run(f'gcloud storage cp "{GCS_OUTPUT}/oof_predictions_roberta_classifier_v2.json" "{MODEL_DIR}/"')

    # 4. Fetch Vertex AI Training Logs
    print(f"\n[4/6] Fetching Vertex AI training logs...")
    job_id = get_latest_job_id()
    if not job_id:
        print("  Warning: Could not automatically detect latest job ID. Using fallback...")
        job_id = "1610858184791031808"
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
        print(f"  [Warning] Failed to fetch training logs. Return code: {log_result.returncode}")
        if log_result.stderr:
            print(f"  STDERR: {log_result.stderr.strip()}")

    # 5. Download model weights (optional)
    if not args.skip_weights:
        print(f"\n[5/6] Downloading model weights for all folds to {MODEL_DIR}/...")
        for i in range(5):
            print(f"  Downloading Fold {i} model and tokenizer...")
            run(f'gcloud storage cp -r "{GCS_OUTPUT}/fold_{i}" "{MODEL_DIR}/"')
        print("  [OK] Model weights downloaded.")
    else:
        print(f"\n[5/6] Skipping model weights download as requested.")

    # 6. Generate confusion matrix and binned classification metrics
    print(f"\n[6/6] Generating confusion matrix & classification metrics from OOF predictions...")
    oof_file = MODEL_DIR / "oof_predictions_roberta_classifier_v2.json"
    if oof_file.exists():
        with open(oof_file) as f:
            oof_data = json.load(f)
        
        all_true_labels = []
        all_pred_labels = []
        for fold_item in oof_data["folds"]:
            all_true_labels.extend(fold_item["true_labels"])
            all_pred_labels.extend(fold_item["pred_labels"])
        
        # Convert label IDs back to class names for display and report
        severity_classes = ['Non-Grievance', 'Low', 'Medium', 'High', 'Critical']
        all_true_cats = [severity_classes[int(l)] for l in all_true_labels]
        all_pred_cats = [severity_classes[int(l)] for l in all_pred_labels]
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_true_cats, all_pred_cats, labels=severity_classes)
        cm_path = MODEL_DIR / "confusion_matrix_roberta.npy"
        np.save(cm_path, cm)
        print(f"  Saved confusion matrix array to: {cm_path}")
        
        # Calculate overall classification metrics
        acc = accuracy_score(all_true_cats, all_pred_cats)
        f1_macro = f1_score(all_true_cats, all_pred_cats, average="macro", zero_division=0)
        f1_weighted = f1_score(all_true_cats, all_pred_cats, average="weighted", zero_division=0)
        prec = precision_score(all_true_cats, all_pred_cats, average="macro", zero_division=0)
        rec = recall_score(all_true_cats, all_pred_cats, average="macro", zero_division=0)
        
        binned_metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": prec,
            "recall_macro": rec
        }
        
        with open(MODEL_DIR / "results_roberta_binned.json", "w") as f:
            json.dump(binned_metrics, f, indent=4)
        
        # Plot confusion matrix
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='PuBuGn', xticklabels=severity_classes, yticklabels=severity_classes,
                    annot_kws={"size": 11, "weight": "bold"}, cbar=True)
        plt.title('RoBERTa Classifier (v2) - Binned Confusion Matrix (OOF)', fontsize=14, pad=15)
        plt.xlabel('Predicted Severity Class', fontsize=12, labelpad=10)
        plt.ylabel('True Severity Class', fontsize=12, labelpad=10)
        plt.xticks(rotation=15)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plot_path = MODEL_DIR / "confusion_matrix_roberta.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved confusion matrix heatmap plot to: {plot_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"  ROBERTA SEVERITY CLASSIFIER BINNED SUMMARY")
        print(f"{'='*60}")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1-Macro:    {f1_macro:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"\n  Classification Report:")
        report = classification_report(all_true_cats, all_pred_cats, labels=severity_classes, target_names=severity_classes, zero_division=0)
        print(report)
        
        # Save classification report to text file
        with open(MODEL_DIR / "classification_report_binned.txt", "w") as f:
            f.write(report)
            
        print(f"{'='*60}")

    print("\n[OK] Script completed successfully!")

if __name__ == "__main__":
    main()
