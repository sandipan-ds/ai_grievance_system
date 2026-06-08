"""
BERT fine-tuning for civic agency classification (8 classes).
Runs on Vertex AI with GPU — 5-fold cross-validation.

This script is uploaded to GCS and executed by the Vertex AI training job.
"""
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PJRT_DEVICE"] = "CUDA"

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "bert-base-uncased"
MAX_SEQ_LEN      = 256
BATCH_SIZE        = 16
LEARNING_RATE     = 2e-5
WEIGHT_DECAY      = 0.01
NUM_EPOCHS        = 200
WARMUP_RATIO      = 0.1
SEED              = 43

TASK_NAME = "civic_agency"


# ── Dataset ───────────────────────────────────────────────────────────────────
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.label2id  = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.label2id[self.labels[idx]]
        enc   = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_fold(fold_idx, train_X, train_y, val_X, val_y, labels, device, output_dir):
    """Train BERT for one fold and return metrics."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}")
    print(f"{'='*60}")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)

    # Tokenizer + model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # Datasets
    train_ds = ComplaintDataset(train_X, train_y, tokenizer, MAX_SEQ_LEN, label2id)
    val_ds   = ComplaintDataset(val_X,   val_y,   tokenizer, MAX_SEQ_LEN, label2id)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Training
    best_val_f1 = -1
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels_t  = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels_t)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=-1)

            all_train_preds.extend(preds.detach().cpu().numpy())
            all_train_labels.extend(labels_t.cpu().numpy())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                avg_loss = total_loss / (step + 1)
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1}/{NUM_EPOCHS} | Step {step+1}/{len(train_dl)} | "
                      f"Loss: {avg_loss:.4f} | {elapsed:.0f}s")

        avg_train_loss = total_loss / len(train_dl)
        all_train_preds = np.array(all_train_preds)
        all_train_labels = np.array(all_train_labels)

        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        train_f1_weighted = f1_score(all_train_labels, all_train_preds, average="weighted", zero_division=0)
        train_prec = precision_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, average="macro", zero_division=0)

        # Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels_t  = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_t.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        val_acc    = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        val_prec   = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        val_rec    = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n    Epoch {epoch+1} Summary:")
        print(f"      [Train] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1-macro: {train_f1_macro:.4f} | F1-weighted: {train_f1_weighted:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f}")
        print(f"      [Val]   Acc: {val_acc:.4f} | F1-macro: {val_f1_macro:.4f} | F1-weighted: {val_f1_weighted:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        print(f"      [LR]    {current_lr:.6f}\n")

        # Step LR scheduler with validation F1-macro
        scheduler.step(val_f1_macro)

        # Save best model / Early stopping check
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    ✅ New best Val F1-macro: {val_f1_macro:.4f}")
        else:
            patience_counter += 1
            print(f"    patience_counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"    Early stopping triggered after {epoch+1} epochs.")
                break

    # Final evaluation with best model
    model.load_state_dict(best_model_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels_t  = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_t.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Convert back to label names
    pred_labels = np.array([id2label[p] for p in all_preds])
    true_labels = np.array([id2label[l] for l in all_labels])

    metrics = {
        "fold": fold_idx,
        "val_accuracy":     float(accuracy_score(all_labels, all_preds)),
        "val_f1_macro":     float(f1_score(all_labels, all_preds, average="macro")),
        "val_f1_micro":     float(f1_score(all_labels, all_preds, average="micro")),
        "val_f1_weighted":  float(f1_score(all_labels, all_preds, average="weighted")),
        "val_precision":    float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "val_recall":       float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "per_class_report": classification_report(true_labels, pred_labels, output_dict=True, zero_division=0),
    }

    # Save fold model
    fold_model_dir = output_dir / f"fold_{fold_idx}"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_model_dir / "model")
    tokenizer.save_pretrained(fold_model_dir / "tokenizer")

    return metrics, true_labels, pred_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to cv_fold_data_final.joblib (local or downloaded from GCS)")
    parser.add_argument("--output-dir", type=str, default="/tmp/bert_civic_output",
                        help="Directory to save results")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading fold data from: {args.data_path}")
    fold_data = joblib.load(args.data_path)
    print(f"Loaded {len(fold_data)} folds")

    # Determine labels
    all_labels_set = set()
    for d in fold_data:
        all_labels_set.update(d["train_y"])
        all_labels_set.update(d["val_y"])
    labels = sorted(all_labels_set)
    print(f"Classes ({len(labels)}): {labels}")

    # Run all folds
    all_metrics = []
    all_true, all_pred = [], []

    for fold_idx, d in enumerate(fold_data):
        metrics, true_labels, pred_labels = train_one_fold(
            fold_idx=fold_idx,
            train_X=d["train_X"],
            train_y=d["train_y"],
            val_X=d["val_X"],
            val_y=d["val_y"],
            labels=labels,
            device=device,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)
        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

    # ── Aggregate results ─────────────────────────────────────────────────────
    avg_metrics = {
        "task":             TASK_NAME,
        "model":            MODEL_NAME,
        "max_seq_len":      MAX_SEQ_LEN,
        "batch_size":       BATCH_SIZE,
        "learning_rate":    LEARNING_RATE,
        "num_epochs":       NUM_EPOCHS,
        "num_folds":        len(fold_data),
        "avg_val_accuracy":    float(np.mean([m["val_accuracy"]    for m in all_metrics])),
        "avg_val_f1_macro":    float(np.mean([m["val_f1_macro"]    for m in all_metrics])),
        "avg_val_f1_micro":    float(np.mean([m["val_f1_micro"]    for m in all_metrics])),
        "avg_val_f1_weighted": float(np.mean([m["val_f1_weighted"] for m in all_metrics])),
        "avg_val_precision":   float(np.mean([m["val_precision"]   for m in all_metrics])),
        "avg_val_recall":      float(np.mean([m["val_recall"]      for m in all_metrics])),
        "per_fold_metrics":    all_metrics,
    }

    # OOF confusion matrix
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=labels)

    # Save results
    results_path = output_dir / "results_bert_civic.json"
    with open(results_path, "w") as f:
        json.dump(avg_metrics, f, indent=4, default=str)
    print(f"\n✅ Results saved: {results_path}")

    # Save confusion matrix as numpy
    cm_path = output_dir / "confusion_matrix_bert_civic.npy"
    np.save(cm_path, cm)
    print(f"✅ Confusion matrix saved: {cm_path}")

    # Save OOF predictions
    oof_path = output_dir / "oof_predictions_civic.joblib"
    joblib.dump({"true": all_true, "pred": all_pred, "labels": labels}, oof_path)
    print(f"✅ OOF predictions saved: {oof_path}")

    # Calculate overall OOF metrics
    oof_acc = accuracy_score(all_true, all_pred)
    oof_f1_macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    oof_f1_weighted = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    oof_prec = precision_score(all_true, all_pred, average="macro", zero_division=0)
    oof_rec = recall_score(all_true, all_pred, average="macro", zero_division=0)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BERT CIVIC AGENCY — FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Avg Val Metrics (across 5 folds):")
    print(f"    Accuracy:     {avg_metrics['avg_val_accuracy']:.4f}")
    print(f"    F1-macro:     {avg_metrics['avg_val_f1_macro']:.4f}")
    print(f"    F1-weighted:  {avg_metrics['avg_val_f1_weighted']:.4f}")
    print(f"    Precision:    {avg_metrics['avg_val_precision']:.4f}")
    print(f"    Recall:       {avg_metrics['avg_val_recall']:.4f}")
    print(f"\n  Overall Out-of-Fold (OOF) Metrics:")
    print(f"    Accuracy:     {oof_acc:.4f}")
    print(f"    F1-macro:     {oof_f1_macro:.4f}")
    print(f"    F1-weighted:  {oof_f1_weighted:.4f}")
    print(f"    Precision:    {oof_prec:.4f}")
    print(f"    Recall:       {oof_rec:.4f}")
    print(f"\n  Per Civic Agency Metrics (Overall OOF):")
    print(classification_report(all_true, all_pred, labels=labels, target_names=labels, zero_division=0))
    print(f"{'='*60}\n")

    # Copy results to AIP_MODEL_DIR for Vertex AI to pick up
    aip_model_dir = os.environ.get("AIP_MODEL_DIR")
    if aip_model_dir:
        import shutil
        print(f"\nCopying results to AIP_MODEL_DIR: {aip_model_dir}")
        
        copied_successfully = False
        
        # Method 1: Try using FUSE mount if it's a gs:// path
        if aip_model_dir.startswith("gs://"):
            local_model_dir = aip_model_dir.replace("gs://", "/gcs/", 1)
            try:
                local_model_path = Path(local_model_dir)
                local_model_path.mkdir(parents=True, exist_ok=True)
                for f in output_dir.iterdir():
                    if f.is_file():
                        dst = local_model_path / f.name
                        shutil.copy2(f, dst)
                        print(f"  Copied via FUSE: {f.name}")
                copied_successfully = True
            except Exception as e:
                print(f"Failed to copy via FUSE path ({local_model_dir}): {e}")
                
        # Method 2: Standard local copy (if aip_model_dir is already a local path)
        else:
            try:
                local_model_path = Path(aip_model_dir)
                local_model_path.mkdir(parents=True, exist_ok=True)
                for f in output_dir.iterdir():
                    if f.is_file():
                        dst = local_model_path / f.name
                        shutil.copy2(f, dst)
                        print(f"  Copied: {f.name}")
                copied_successfully = True
            except Exception as e:
                print(f"Failed to copy locally: {e}")

        # Method 3: Fallback to gsutil if the previous methods failed and it's a gs:// path
        if not copied_successfully and aip_model_dir.startswith("gs://"):
            print("Attempting fallback copy via gsutil...")
            try:
                import subprocess
                for f in output_dir.iterdir():
                    if f.is_file():
                        gcs_dst = f"{aip_model_dir.rstrip('/')}/{f.name}"
                        subprocess.run(["gsutil", "cp", str(f), gcs_dst], check=True)
                        print(f"  Copied via gsutil: {f.name}")
                copied_successfully = True
            except Exception as e:
                print(f"Failed to copy via gsutil: {e}")


if __name__ == "__main__":
    main()
