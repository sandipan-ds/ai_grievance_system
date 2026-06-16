"""
Submit Tier A severity regression job (XGBoost + sentence-transformer embeddings)
to Vertex AI Custom Training.

Usage: python scripts/submit_vertex_job_severity_xgb.py
"""
import subprocess
from pathlib import Path
from google.cloud import aiplatform
import time

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "ai-grievance-sandi"
REGION       = "us-central1"
BUCKET_NAME  = f"{PROJECT_ID}-data"
BUCKET_URI   = f"gs://{BUCKET_NAME}"

GCS_DATA_DIR       = f"{BUCKET_URI}/bert_training/severity_v2"
GCS_OUTPUT         = f"{GCS_DATA_DIR}/output_xgb"
GCS_CHECKPOINT_DIR = f"{GCS_DATA_DIR}/checkpoints_xgb"

# The dataset is already uploaded under civic_v2 — reuse it
GCS_DATA_FILE = f"{BUCKET_URI}/bert_training/civic_v2/df_final_nlp_bert_v2.joblib"

# Deep Learning Container (PyTorch GPU)
TRAIN_IMAGE = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310:latest"

JOB_DISPLAY_NAME = f"xgb-severity-regressor-{int(time.time())}"


def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # ── Upload the training script to GCS ─────────────────────────────────────
    local_script = Path(__file__).resolve().parent / "xgb_severity_regressor.py"
    gcs_script_dest = f"{GCS_DATA_DIR}/xgb_severity_regressor.py"

    print(f"Uploading training script to GCS: {gcs_script_dest}")
    cmd = f'gcloud storage cp "{local_script}" {gcs_script_dest}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Error uploading script: {result.stderr.strip()}")
    else:
        print(f"  [OK] Script uploaded.")

    print(f"\nSubmitting Tier A severity regression job: {JOB_DISPLAY_NAME}")
    print(f"  Data:   {GCS_DATA_FILE}")
    print(f"  Output: {GCS_OUTPUT}")
    print(f"  Image:  {TRAIN_IMAGE}")
    print(f"  Machine: n1-standard-8 + 1x T4 GPU (~$1.20/hr)")
    print(f"  Models:  Ridge, Decision Tree, Random Forest, XGBoost (GPU), LightGBM")

    job = aiplatform.CustomJob(
        display_name=JOB_DISPLAY_NAME,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAIN_IMAGE,
                "command": ["bash", "-c"],
                "args": [
                    "pip install --upgrade "
                    "'transformers>=4.40,<4.45' && "
                    "pip install --upgrade "
                    "sentence-transformers xgboost lightgbm "
                    "scikit-learn joblib numpy pandas && "
                    "python -u /gcs/ai-grievance-sandi-data/bert_training/severity_v2/xgb_severity_regressor.py "
                    f"--data-path /gcs/ai-grievance-sandi-data/bert_training/civic_v2/df_final_nlp_bert_v2.joblib "
                    f"--output-dir /gcs/ai-grievance-sandi-data/bert_training/severity_v2/output_xgb "
                    f"--checkpoint-dir /gcs/ai-grievance-sandi-data/bert_training/severity_v2/checkpoints_xgb"
                ],
            },
        }],
        base_output_dir=GCS_OUTPUT,
    )

    print(f"\n  Submitting job... (this will block until the job finishes)")
    print(f"  Once you see 'JOB_STATE_RUNNING' in the console, you can Ctrl+C safely.\n")

    job.run(sync=True)

    print(f"\n[OK] Job completed!")
    print(f"  Job name: {job.display_name}")
    print(f"  State:    {job.state}")
    print(f"\n  Monitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    main()
