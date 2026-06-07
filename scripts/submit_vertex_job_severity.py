"""
Submit DistilBERT severity classification training job to Vertex AI.
Usage: python scripts/submit_vertex_job_severity.py
"""
from google.cloud import aiplatform
import time

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID   = "ai-grievance-sandi"
REGION       = "us-central1"
BUCKET_NAME  = f"{PROJECT_ID}-data"
BUCKET_URI   = f"gs://{BUCKET_NAME}"

GCS_DATA_DIR = f"{BUCKET_URI}/bert_training/severity"
GCS_OUTPUT   = f"{GCS_DATA_DIR}/output"
GCS_CHECKPOINT_DIR = f"{GCS_DATA_DIR}/checkpoints"

# Deep Learning Container with PyTorch GPU support
TRAIN_IMAGE = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310:latest"

JOB_DISPLAY_NAME = f"distilbert-severity-{int(time.time())}"

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    print(f"Submitting DistilBERT training job: {JOB_DISPLAY_NAME}")
    print(f"  Data:   {GCS_DATA_DIR}/cv_fold_data_final.joblib")
    print(f"  Output: {GCS_OUTPUT}")
    print(f"  Checkpoints: {GCS_CHECKPOINT_DIR}")
    print(f"  Image:  {TRAIN_IMAGE}")
    print(f"  Machine: n1-standard-4 + 1x T4 GPU (~$0.95/hr)")

    # Define custom job worker specs
    job = aiplatform.CustomJob(
        display_name=JOB_DISPLAY_NAME,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAIN_IMAGE,
                "command": ["bash", "-c"],
                "args": [
                    "export PJRT_DEVICE=CUDA && "
                    "pip install --upgrade joblib scikit-learn pandas && "
                    "pip install 'transformers>=4.40,<4.45' && "
                    "python -u /gcs/ai-grievance-sandi-data/bert_training/severity/distilbert_train_severity.py "
                    "--data-path /gcs/ai-grievance-sandi-data/bert_training/civic/cv_fold_data_final.joblib "
                    "--output-dir /gcs/ai-grievance-sandi-data/bert_training/severity/output "

                    "--checkpoint-dir /gcs/ai-grievance-sandi-data/bert_training/severity/checkpoints"
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
