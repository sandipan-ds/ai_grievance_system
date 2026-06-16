"""
Extract Softmax Probabilities from Saved Fold Models
=====================================================
Instead of re-training 3 models from scratch, this script:
  1. Loads the saved fold model weights from GCS (already trained)
  2. Runs forward passes on the validation data for each fold
  3. Extracts softmax probability vectors
  4. Saves updated OOF files with 'pred_probs' key

Runs as a SINGLE Vertex AI job (~30 min) instead of 3 separate jobs (~6 hrs).
All 3 models (DistilBERT, RoBERTa, DeBERTa v3) are processed sequentially.

Usage (Vertex AI):
  python extract_probs_civic.py \
    --data-path gs://bucket/path/to/fold_data.joblib \
    --output-dir gs://bucket/path/to/stacking_output
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PJRT_DEVICE"] = "CUDA"

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
MAX_SEQ_LEN = 256
BATCH_SIZE = 16

# Model configurations: (name, HuggingFace model ID, GCS output dir with fold weights)
MODELS = [
    {
        "name": "DistilBERT",
        "hf_model": "distilbert-base-uncased",
        "gcs_output": "gs://ai-grievance-sandi-data/bert_training/civic_v2/output",
        "oof_filename": "oof_predictions_civic_distilbert_probs.joblib",
    },
    {
        "name": "RoBERTa",
        "hf_model": "roberta-base",
        "gcs_output": "gs://ai-grievance-sandi-data/bert_training/civic_v2/roberta_output",
        "oof_filename": "oof_predictions_civic_roberta_probs.joblib",
    },
    {
        "name": "DeBERTa v3",
        "hf_model": "microsoft/deberta-v3-base",
        "gcs_output": "gs://ai-grievance-sandi-data/bert_training/civic_v2/deberta_v3_output",
        "oof_filename": "oof_predictions_civic_deberta_v3_probs.joblib",
    },
]


class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label2id[self.labels[idx]]
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def resolve_path(p):
    """Convert gs:// paths to /gcs/ mount paths on Vertex AI."""
    if p.startswith("gs://"):
        return Path(p.replace("gs://", "/gcs/", 1))
    return Path(p)


def extract_probs_for_model(model_config, fold_data, labels, device, output_dir):
    """Load saved fold models and extract softmax probabilities for one model."""
    model_name = model_config["name"]
    hf_model = model_config["hf_model"]
    gcs_output = resolve_path(model_config["gcs_output"])

    print(f"\n{'='*60}")
    print(f"  EXTRACTING PROBABILITIES: {model_name}")
    print(f"{'='*60}")
    print(f"  HF Model: {hf_model}")
    print(f"  Fold weights dir: {gcs_output}")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)

    all_true = []
    all_pred = []
    all_probs = []
    fold_latencies = []

    for fold_idx, d in enumerate(fold_data):
        print(f"\n  --- Fold {fold_idx} ---")

        # Load saved model for this fold
        fold_model_dir = gcs_output / f"fold_{fold_idx}" / "model"
        if not fold_model_dir.exists():
            print(f"  ERROR: Fold model not found at {fold_model_dir}")
            print(f"  Checking alternative path...")
            # Try without /model subdirectory
            fold_model_dir = gcs_output / f"fold_{fold_idx}"
            if not fold_model_dir.exists():
                raise FileNotFoundError(f"Cannot find fold {fold_idx} model at {gcs_output}")

        print(f"  Loading model from: {fold_model_dir}")

        # Use the model-specific tokenizer if saved, otherwise use HF default
        tokenizer_dir = gcs_output / f"fold_{fold_idx}" / "tokenizer"
        if tokenizer_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            str(fold_model_dir),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        ).to(device)
        model.eval()

        # Create validation dataloader
        val_ds = ComplaintDataset(d["val_X"], d["val_y"], tokenizer, MAX_SEQ_LEN, label2id)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # Forward pass to extract probabilities
        fold_preds = []
        fold_probs = []
        fold_labels = []

        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels_t = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                fold_preds.extend(preds.cpu().numpy())
                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels_t.cpu().numpy())

        fold_preds = np.array(fold_preds)
        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)

        pred_labels = np.array([id2label[p] for p in fold_preds])
        true_labels = np.array([id2label[l] for l in fold_labels])

        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(fold_labels, fold_preds)
        f1 = f1_score(fold_labels, fold_preds, average="macro", zero_division=0)
        print(f"  Fold {fold_idx}: Acc={acc:.4f}, F1-Macro={f1:.4f}, Probs shape={fold_probs.shape}")

        all_true.extend(true_labels)
        all_pred.extend(pred_labels)
        all_probs.extend(fold_probs)

        # Latency benchmark (100 single-sample inferences)
        sample_batch = next(iter(val_dl))
        sample_input = {k: v[:1].to(device) for k, v in sample_batch.items() if k != "label"}
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                t_start = time.time()
                _ = model(**sample_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.time() - t_start) * 1000)
        avg_latency = float(np.mean(latencies))
        fold_latencies.append(avg_latency)
        print(f"  Latency: {avg_latency:.2f} ms ({device})")

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_probs = np.array(all_probs)

    # Save OOF predictions with probabilities
    oof_path = output_dir / model_config["oof_filename"]
    joblib.dump({
        "true": all_true,
        "pred": all_pred,
        "pred_probs": all_probs,
        "labels": labels,
        "avg_inference_latency_ms": float(np.mean(fold_latencies)),
        "inference_device": str(device),
    }, oof_path)

    from sklearn.metrics import accuracy_score, f1_score
    overall_acc = accuracy_score(all_true, all_pred)
    overall_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    print(f"\n  {model_name} Overall: Acc={overall_acc:.4f}, F1-Macro={overall_f1:.4f}")
    print(f"  Avg Latency: {np.mean(fold_latencies):.2f} ms")
    print(f"  OOF saved: {oof_path} (shape: {all_probs.shape})")

    return {
        "model": model_name,
        "accuracy": float(overall_acc),
        "f1_macro": float(overall_f1),
        "avg_latency_ms": float(np.mean(fold_latencies)),
        "probs_shape": list(all_probs.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to fold data joblib")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save probability OOF files")
    args = parser.parse_args()

    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fold data
    print(f"\nLoading fold data from: {data_path}")
    fold_data = joblib.load(data_path)
    print(f"Loaded {len(fold_data)} folds")

    # Determine labels
    all_labels_set = set()
    for d in fold_data:
        all_labels_set.update(d["train_y"])
        all_labels_set.update(d["val_y"])
    labels = sorted(all_labels_set)
    print(f"Classes ({len(labels)}): {labels}")

    # Process each model
    results = []
    for model_config in MODELS:
        result = extract_probs_for_model(model_config, fold_data, labels, device, output_dir)
        results.append(result)

    # Save summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"models": results, "device": str(device)}, f, indent=4)
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"  ALL PROBABILITY EXTRACTIONS COMPLETE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']:15s}  Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  Latency={r['avg_latency_ms']:.2f}ms")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
