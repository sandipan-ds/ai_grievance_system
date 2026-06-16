"""
DistilBERT fine-tuning for civic agency classification.
Runs on Vertex AI with GPU — 5-fold cross-validation.
Supports fold-level and epoch-level resuming from Cloud Storage checkpoints.
"""
import os
import sys
import shutil
import random
import argparse
import json
import time
from pathlib import Path

# Force line buffering so logs show up immediately in Google Cloud Console
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PJRT_DEVICE"] = "CUDA"

import joblib
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
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
MODEL_NAME        = "distilbert-base-uncased"
MAX_SEQ_LEN       = 512
BATCH_SIZE        = 32
LEARNING_RATE     = 2e-5
WEIGHT_DECAY      = 0.01
NUM_EPOCHS        = 200
SEED              = 43
TASK_NAME         = "civic_agency"


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


def train_one_fold(fold_idx, train_X, train_y, val_X, val_y, labels, device, checkpoint_dir, output_dir):
    """Train DistilBERT for one fold supporting checkpoint resume."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx} — TRAINING CIVIC AGENCY")
    print(f"{'='*60}")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)

    checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_checkpoint.pt"
    completed_path  = checkpoint_dir / f"fold_{fold_idx}_completed.json"

    # 1. Check if fold already completed
    if completed_path.exists():
        print(f"✅ Fold {fold_idx} is already completed. Loading metrics and predictions from cache...")
        with open(completed_path, "r") as f:
            completed_data = json.load(f)
        pred_probs = np.array(completed_data["pred_probs"]) if "pred_probs" in completed_data else None
        return (
            completed_data["metrics"],
            np.array(completed_data["true_labels"]),
            np.array(completed_data["pred_labels"]),
            pred_probs,
        )

    # 2. Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # 4. Datasets & Dataloaders
    train_ds = ComplaintDataset(train_X, train_y, tokenizer, MAX_SEQ_LEN, label2id)
    val_ds   = ComplaintDataset(val_X,   val_y,   tokenizer, MAX_SEQ_LEN, label2id)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # ReduceLR on plateau after 3 epochs of no improvement (patience=2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # 6. Initialize tracking variables
    start_epoch = 0
    best_val_f1_macro = -1
    best_model_state = None
    patience = 5
    patience_counter = 0

    # 7. Check if checkpoint exists to resume
    if checkpoint_path.exists():
        print(f"🔄 Checkpoint found at {checkpoint_path}. Restoring state...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_f1_macro = checkpoint["best_val_f1_macro"]
            patience_counter = checkpoint["patience_counter"]
            best_model_state = checkpoint["best_model_state"]
            
            # Restore RNG states
            torch.set_rng_state(checkpoint["rng_state"].cpu())
            if checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
            np.random.set_state(checkpoint["np_rng_state"])
            random.setstate(checkpoint["random_rng_state"])
            
            print(f"🔄 Resumed training from Epoch {start_epoch + 1}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Starting training from scratch.")

    # 8. Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
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

        # 9. Compute train metrics
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1_macro = f1_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        train_f1_micro = f1_score(all_train_labels, all_train_preds, average="micro", zero_division=0)
        train_f1_weighted = f1_score(all_train_labels, all_train_preds, average="weighted", zero_division=0)
        train_prec = precision_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, average="macro", zero_division=0)

        # 10. Validation loop
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

        # 11. Compute validation metrics
        val_acc    = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        val_f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        val_prec   = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        val_rec    = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n    Epoch {epoch+1} Summary:")
        print(f"      [Train] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1-macro: {train_f1_macro:.4f} | F1-micro: {train_f1_micro:.4f} | F1-weighted: {train_f1_weighted:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f}")
        print(f"      [Val]   Acc: {val_acc:.4f} | F1-macro: {val_f1_macro:.4f} | F1-micro: {val_f1_micro:.4f} | F1-weighted: {val_f1_weighted:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        print(f"      [LR]    {current_lr:.6f}\n")

        # Step LR scheduler with validation F1-macro
        scheduler.step(val_f1_macro)

        # Save best model / Early stopping check
        is_best = False
        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            is_best = True
            print(f"    ✅ New best Val F1-macro: {val_f1_macro:.4f}")
        else:
            patience_counter += 1
            print(f"    patience_counter: {patience_counter}/{patience}")

        # Save checkpoint to GCS (write locally first, then copy to prevent corruption)
        print(f"    Saving epoch checkpoint...")
        local_ckpt_temp = Path("/tmp") / f"checkpoint_fold_{fold_idx}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1_macro": best_val_f1_macro,
            "patience_counter": patience_counter,
            "best_model_state": best_model_state,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "np_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
        }, local_ckpt_temp)
        shutil.copy2(local_ckpt_temp, checkpoint_path)
        print(f"    Checkpoint saved to {checkpoint_path}")

        if patience_counter >= patience:
            print(f"    Early stopping triggered after {epoch+1} epochs.")
            break

    # 12. Final evaluation with best model state
    model.load_state_dict(best_model_state)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels_t  = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels_t.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)  # shape: (n_val, n_classes)

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

    # Save final model checkpoints to output directory
    fold_model_dir = output_dir / f"fold_{fold_idx}"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_model_dir / "model")
    tokenizer.save_pretrained(fold_model_dir / "tokenizer")

    # 13. Latency benchmark (100 single-sample inferences)
    print(f"    Running latency benchmark (100 samples)...")
    sample_batch = next(iter(val_dl))
    sample_input = {k: v[:1].to(device) for k, v in sample_batch.items() if k != "label"}
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t_start = time.time()
            _ = model(**sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.time() - t_start) * 1000)  # ms
    avg_latency_ms = float(np.mean(latencies))
    metrics["avg_inference_latency_ms"] = avg_latency_ms
    metrics["inference_device"] = str(device)
    print(f"    Avg inference latency: {avg_latency_ms:.2f} ms ({device})")

    # Write completed fold sentinel file
    completed_data = {
        "metrics": metrics,
        "true_labels": true_labels.tolist(),
        "pred_labels": pred_labels.tolist(),
        "pred_probs": all_probs.tolist(),
    }
    local_comp_temp = Path("/tmp") / f"completed_fold_{fold_idx}.json"
    with open(local_comp_temp, "w") as f:
        json.dump(completed_data, f)
    shutil.copy2(local_comp_temp, completed_path)
    print(f"✅ Fold {fold_idx} completed and logged to {completed_path}")

    # Remove temporary epoch checkpoint file
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return metrics, true_labels, pred_labels, all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to fold data joblib")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save final results")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory to save checkpoints for resume")
    args = parser.parse_args()

    # Seed all
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Resolve paths (supporting gs:// paths by replacing with /gcs/)
    def resolve_path(p):
        if p.startswith("gs://"):
            return Path(p.replace("gs://", "/gcs/", 1))
        return Path(p)

    output_dir = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    data_path = resolve_path(args.data_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading fold data from: {data_path}")
    fold_data = joblib.load(data_path)
    print(f"Loaded {len(fold_data)} folds")

    # Determine unique classes (mapping civic agency labels)
    all_labels_set = set()
    for d in fold_data:
        all_labels_set.update(d["train_y"])
        all_labels_set.update(d["val_y"])
    labels = sorted(all_labels_set)
    print(f"Classes ({len(labels)}): {labels}")

    # Run folds
    all_metrics = []
    all_true, all_pred, all_probs = [], [], []

    for fold_idx, d in enumerate(fold_data):
        metrics, true_labels, pred_labels, pred_probs = train_one_fold(
            fold_idx=fold_idx,
            train_X=d["train_X"],
            train_y=d["train_y"], # Map target feature: civic agency (train_y)
            val_X=d["val_X"],
            val_y=d["val_y"],   # Map target feature: civic agency (val_y)
            labels=labels,
            device=device,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)
        all_true.extend(true_labels)
        all_pred.extend(pred_labels)
        if pred_probs is not None:
            all_probs.extend(pred_probs)

    # ── Aggregate Results ─────────────────────────────────────────────────────
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_true, all_pred, labels=labels)

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

    # Save results
    results_path = output_dir / "results_distilbert_civic.json"
    with open(results_path, "w") as f:
        json.dump(avg_metrics, f, indent=4, default=str)
    print(f"\n✅ Results saved: {results_path}")

    # Save confusion matrix array
    cm_path = output_dir / "confusion_matrix_distilbert_civic.npy"
    np.save(cm_path, cm)
    print(f"✅ Confusion matrix saved: {cm_path}")

    # Save OOF predictions
    oof_path = output_dir / "oof_predictions_civic.joblib"
    oof_data = {"true": all_true, "pred": all_pred, "labels": labels}
    if all_probs:
        oof_data["pred_probs"] = np.array(all_probs)
    joblib.dump(oof_data, oof_path)
    print(f"✅ OOF predictions saved: {oof_path}")
    if all_probs:
        print(f"   (includes probability vectors: shape {np.array(all_probs).shape})")

    # Calculate overall OOF metrics
    oof_acc = accuracy_score(all_true, all_pred)
    oof_f1_macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    oof_f1_micro = f1_score(all_true, all_pred, average="micro", zero_division=0)
    oof_f1_weighted = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    oof_prec = precision_score(all_true, all_pred, average="macro", zero_division=0)
    oof_rec = recall_score(all_true, all_pred, average="macro", zero_division=0)

    # ── Final Metric Output ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DISTILBERT CIVIC AGENCY — FINAL TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"  Average Validation Metrics (across 5 folds):")
    print(f"    Accuracy:     {avg_metrics['avg_val_accuracy']:.4f}")
    print(f"    F1-macro:     {avg_metrics['avg_val_f1_macro']:.4f}")
    print(f"    F1-micro:     {avg_metrics['avg_val_f1_micro']:.4f}")
    print(f"    F1-weighted:  {avg_metrics['avg_val_f1_weighted']:.4f}")
    print(f"    Precision:    {avg_metrics['avg_val_precision']:.4f}")
    print(f"    Recall:       {avg_metrics['avg_val_recall']:.4f}")
    
    print(f"\n  Overall Out-of-Fold (OOF) Metrics:")
    print(f"    Accuracy:     {oof_acc:.4f}")
    print(f"    F1-macro:     {oof_f1_macro:.4f}")
    print(f"    F1-micro:     {oof_f1_micro:.4f}")
    print(f"    F1-weighted:  {oof_f1_weighted:.4f}")
    print(f"    Precision:    {oof_prec:.4f}")
    print(f"    Recall:       {oof_rec:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"    Classes: {labels}")
    print(cm)
    
    print(f"\n  Per Civic Agency Metrics (Overall OOF):")
    print(classification_report(all_true, all_pred, labels=labels, target_names=labels, zero_division=0))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
