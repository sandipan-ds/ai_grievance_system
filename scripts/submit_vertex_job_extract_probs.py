"""
Submit a SINGLE Vertex AI job to extract softmax probabilities from
all 3 saved models (DistilBERT, RoBERTa, DeBERTa v3).

This is inference-only (~30 min), NOT re-training (~6 hrs).

Usage: python scripts/submit_vertex_job_extract_probs.py
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

GCS_DATA_DIR = f"{BUCKET_URI}/bert_training/civic_v2"
GCS_OUTPUT   = f"{GCS_DATA_DIR}/stacking_probs_output"

# Deep Learning Container with GPU support
TRAIN_IMAGE = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310:latest"

JOB_DISPLAY_NAME = f"extract-probs-civic-stacking-{int(time.time())}"


def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    # ── Upload the extraction script ──────────────────────────────────────────
    local_script = Path(__file__).resolve().parent / "extract_probs_civic.py"
    gcs_script_dest = f"{GCS_DATA_DIR}/extract_probs_civic.py"

    print(f"Uploading extraction script to GCS: {gcs_script_dest}")
    cmd = f'gcloud storage cp "{local_script}" {gcs_script_dest}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Error uploading script: {result.stderr.strip()}")
    else:
        print(f"  [OK] Script uploaded.")

    print(f"\nSubmitting probability extraction job: {JOB_DISPLAY_NAME}")
    print(f"  Data:   {GCS_DATA_DIR}/df_final_nlp_bert_v2.joblib")
    print(f"  Output: {GCS_OUTPUT}")
    print(f"  Image:  {TRAIN_IMAGE}")
    print(f"  Machine: n1-standard-4 + 1x T4 GPU (~$0.95/hr)")
    print(f"\n  This job loads saved fold weights from:")
    print(f"    - {GCS_DATA_DIR}/output/fold_*/model/          (DistilBERT)")
    print(f"    - {GCS_DATA_DIR}/roberta_output/fold_*/model/  (RoBERTa)")
    print(f"    - {GCS_DATA_DIR}/deberta_v3_output/fold_*/model/ (DeBERTa v3)")

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
                    "pip install --upgrade joblib scikit-learn pandas numpy sentencepiece protobuf && "
                    "pip install 'transformers>=4.40,<4.45' && "
                    "python -u /gcs/ai-grievance-sandi-data/bert_training/civic_v2/extract_probs_civic.py "
                    "--data-path /gcs/ai-grievance-sandi-data/bert_training/civic_v2/df_final_nlp_bert_v2.joblib "
                    "--output-dir /gcs/ai-grievance-sandi-data/bert_training/civic_v2/stacking_probs_output"
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
    print(f"\n  After completion, run:")
    print(f"  python scripts/download_results_extract_probs.py")


if __name__ == "__main__":
    main()
