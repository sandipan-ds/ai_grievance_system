"""
Trial 5: Fine-tuned RoBERTa-base classifier for severity classification (v2).
Input:  complaint description text + severity reason (ground-truth for train, T5-generated/ground-truth for val)
Output: binned severity class (0: Non-Grievance, 1: Low, 2: Medium, 3: High, 4: Critical)
Runs on Vertex AI with GPU — 5-fold cross-validation.
"""
import os
import sys
import argparse
import json
import time
import random
import warnings
from pathlib import Path

# Force line buffering so logs show up immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PJRT_DEVICE"] = "CUDA"

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import RobertaModel, RobertaTokenizer
# Suppress the overflowing tokens warnings from transformers tokenizer
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)
from nltk.translate.meteor_score import meteor_score

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "roberta-base"
MAX_INPUT_LEN   = 256
BATCH_SIZE      = 16
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01
NUM_EPOCHS      = 10
SEED            = 43
PATIENCE        = 3

SEVERITY_CLASSES = ['Non-Grievance', 'Low', 'Medium', 'High', 'Critical']

# ── Model ─────────────────────────────────────────────────────────────────────
class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1, num_classes=5):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes) # 768 → 5

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ── Dataset ───────────────────────────────────────────────────────────────────
class SeverityClassificationDataset(Dataset):
    def __init__(self, texts, reasons, labels, tokenizer, max_len):
        self.texts     = texts
        self.reasons    = reasons
        self.labels     = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        reason = str(self.reasons[idx])
        label = int(self.labels[idx])

        # Tokenize pair inputs correctly using RoBERTa's double-segment format
        encoding = self.tokenizer(
            text,
            text_pair=reason,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── Text Metrics ──────────────────────────────────────────────────────────────
def compute_rouge_l(pred, ref):
    pred_tokens = str(pred).lower().split()
    ref_tokens  = str(ref).lower().split()
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_meteor(pred, ref):
    pred_tokens = str(pred).lower().split()
    ref_tokens  = str(ref).lower().split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    return float(meteor_score([ref_tokens], pred_tokens))


# ── Evaluation ────────────────────────────────────────────────────────────────
def get_severity_class_id(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 2  # Medium fallback
    if score >= 90:
        return 4  # Critical
    elif score >= 80:
        return 3  # High
    elif score >= 50:
        return 2  # Medium
    elif score >= 1:
        return 1  # Low
    else:
        return 0  # Non-Grievance


def resolve_path(p):
    if p.startswith("gs://"):
        return Path(p.replace("gs://", "/gcs/", 1))
    return Path(p)


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            logits = model(input_ids, attn_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=-1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    acc = accuracy_score(all_true, all_pred)
    
    # Classification Metrics (Macro, Micro, Weighted)
    f1_macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    f1_micro = f1_score(all_true, all_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    
    prec_macro = precision_score(all_true, all_pred, average="macro", zero_division=0)
    prec_micro = precision_score(all_true, all_pred, average="micro", zero_division=0)
    prec_weighted = precision_score(all_true, all_pred, average="weighted", zero_division=0)
    
    rec_macro = recall_score(all_true, all_pred, average="macro", zero_division=0)
    rec_micro = recall_score(all_true, all_pred, average="micro", zero_division=0)
    rec_weighted = recall_score(all_true, all_pred, average="weighted", zero_division=0)
    
    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "precision_micro": float(prec_micro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_micro": float(rec_micro),
        "recall_weighted": float(rec_weighted),
        "true": all_true,
        "pred": all_pred
    }


# ── Training ─────────────────────────────────────────────────────────────────
def train_one_fold(fold_idx, train_X, train_reasons, train_labels,
                   val_X, val_reasons_gt, val_labels,
                   device, checkpoint_dir, output_dir, t5_checkpoints_dir):
    print(f"\n{'=' * 70}")
    print(f"  FOLD {fold_idx} — RoBERTa SEVERITY CLASSIFIER")
    print(f"{'=' * 70}")

    checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_checkpoint.pt"
    completed_path  = checkpoint_dir / f"fold_{fold_idx}_completed.json"

    # ── 1. Check if fold already completed ────────────────────────────────────
    if completed_path.exists():
        print(f"✅ Fold {fold_idx} already completed. Loading from cache...")
        with open(completed_path, "r") as f:
            completed_data = json.load(f)
        return completed_data["metrics"]

    # ── 2. Load T5-Generated Reasons if available ──────────────────────────────
    val_reasons_to_use = list(val_reasons_gt)
    using_predicted_reasons = False
    
    if t5_checkpoints_dir:
        t5_completed_path = Path(t5_checkpoints_dir) / f"fold_{fold_idx}_completed.json"
        if t5_completed_path.exists():
            try:
                with open(t5_completed_path, "r") as f:
                    t5_cdata = json.load(f)
                if len(t5_cdata["pred_reasons"]) == len(val_reasons_gt):
                    val_reasons_to_use = t5_cdata["pred_reasons"]
                    using_predicted_reasons = True
                    print(f"  [OK] Loaded T5-generated validation reasons from: {t5_completed_path}")
                else:
                    print(f"  [Warning] Length mismatch in {t5_completed_path}. Using ground-truth validation reasons.")
            except Exception as e:
                print(f"  [Warning] Failed to load {t5_completed_path}: {e}. Using ground-truth validation reasons.")
        else:
            print(f"  [Info] T5 fold file {t5_completed_path} not found. Using ground-truth validation reasons.")
    else:
        print(f"  [Info] T5 checkpoints directory not provided. Using ground-truth validation reasons.")

    # ── 3. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = RoBERTaClassifier(MODEL_NAME).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # ── 5. Datasets & Dataloaders ─────────────────────────────────────────────
    train_ds = SeverityClassificationDataset(train_X, train_reasons, train_labels, tokenizer, MAX_INPUT_LEN)
    val_ds   = SeverityClassificationDataset(val_X, val_reasons_to_use, val_labels, tokenizer, MAX_INPUT_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Deterministic Train subset for fast train progress evaluation
    train_X_list = list(train_X)
    train_reasons_list = list(train_reasons)
    train_labels_list = list(train_labels)
    eval_train_size = min(500, len(train_X_list))
    rng = random.Random(SEED)
    eval_indices = rng.sample(range(len(train_X_list)), eval_train_size)
    train_X_eval = [train_X_list[i] for i in eval_indices]
    train_reasons_eval = [train_reasons_list[i] for i in eval_indices]
    train_labels_eval = [train_labels_list[i] for i in eval_indices]
    train_eval_ds = SeverityClassificationDataset(train_X_eval, train_reasons_eval, train_labels_eval, tokenizer, MAX_INPUT_LEN)
    train_eval_dl = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Compute Class Weights for Imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    weights_tensor = torch.zeros(5, dtype=torch.float32)
    for c_id, w in zip(np.unique(train_labels), class_weights):
        weights_tensor[c_id] = float(w)
    weights_tensor[weights_tensor == 0.0] = 1.0
    weights_tensor = weights_tensor.to(device)
    print(f"  Class Weights (balanced): {weights_tensor.cpu().numpy().tolist()}")

    # Loss Function, Optimizer & Scheduler
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    # Tracking
    start_epoch       = 0
    best_val_loss     = float("inf")
    best_model_state  = None
    patience_counter  = 0
    epoch_log         = []

    # Resume from checkpoint
    if checkpoint_path.exists():
        print(f"🔄 Checkpoint found. Restoring state...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            start_epoch      = ckpt["epoch"] + 1
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_val_loss    = ckpt["best_val_loss"]
            patience_counter = ckpt["patience_counter"]
            best_model_state = ckpt["best_model_state"]
            epoch_log        = ckpt.get("epoch_log", [])

            torch.set_rng_state(ckpt["rng_state"].cpu())
            if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
            np.random.set_state(ckpt["np_rng_state"])
            random.setstate(ckpt["random_rng_state"])

            print(f"🔄 Resumed from Epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}. Starting fresh.")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            logits = model(input_ids, attn_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"  Fold {fold_idx} | Epoch {epoch + 1}/{NUM_EPOCHS} | "
                      f"Step {step + 1}/{len(train_dl)} | Loss: {loss.item():.4f}")

        elapsed = time.time() - t0

        # Evaluate on train subset & validation set
        train_metrics = evaluate(model, train_eval_dl, device, loss_fn)
        val_metrics   = evaluate(model, val_dl, device, loss_fn)

        # Print all classification metrics
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n  Fold {fold_idx} | Epoch {epoch + 1}/{NUM_EPOCHS} | Time: {elapsed:.0f}s | LR: {current_lr:.2e}")
        print(f"    Train  Loss={train_metrics['loss']:.4f}  Acc={train_metrics['accuracy']:.4f}  F1-Macro={train_metrics['f1_macro']:.4f}")
        print(f"    Val    Loss={val_metrics['loss']:.4f}  Acc={val_metrics['accuracy']:.4f}")
        print(f"    Val Classification Details:")
        print(f"      F1 (Macro/Micro/Weighted):       {val_metrics['f1_macro']:.4f} / {val_metrics['f1_micro']:.4f} / {val_metrics['f1_weighted']:.4f}")
        print(f"      Precision (Macro/Micro/Weighted): {val_metrics['precision_macro']:.4f} / {val_metrics['precision_micro']:.4f} / {val_metrics['precision_weighted']:.4f}")
        print(f"      Recall (Macro/Micro/Weighted):    {val_metrics['recall_macro']:.4f} / {val_metrics['recall_micro']:.4f} / {val_metrics['recall_weighted']:.4f}")

        # Compute ROUGE-L and METEOR if predicted reasons were loaded
        if using_predicted_reasons:
            val_rouges = [compute_rouge_l(p, t) for p, t in zip(val_reasons_to_use, val_reasons_gt)]
            val_meteors = [compute_meteor(p, t) for p, t in zip(val_reasons_to_use, val_reasons_gt)]
            val_rouge = float(np.mean(val_rouges))
            val_meteor = float(np.mean(val_meteors))
            print(f"    T5 reasons metric:  ROUGE-L={val_rouge:.4f}  METEOR={val_meteor:.4f}")
        else:
            print(f"    T5 reasons metric:  N/A (using ground-truth validation reasons)")

        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_micro": val_metrics["f1_micro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "lr": current_lr,
            "elapsed_s": elapsed,
        }
        if using_predicted_reasons:
            epoch_entry["val_rouge_l"] = val_rouge
            epoch_entry["val_meteor"] = val_meteor

        epoch_log.append(epoch_entry)

        # Scheduler step
        scheduler.step(val_metrics["loss"])

        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✅ New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    ⏳ No improvement ({patience_counter}/{PATIENCE})")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "best_model_state": best_model_state,
            "epoch_log": epoch_log,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "np_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
        }, checkpoint_path)

        if patience_counter >= PATIENCE:
            print(f"\n  ⛔ Early stopping triggered at epoch {epoch + 1}")
            break

    # Final evaluation with best model
    print(f"\n  Loading best model state for final evaluation...")
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    val_eval_final = evaluate(model, val_dl, device, loss_fn)
    train_eval_final = evaluate(model, train_eval_dl, device, loss_fn)

    print(f"\n  FOLD {fold_idx} FINAL RESULTS (best checkpoint):")
    print(f"    Train  Acc={train_eval_final['accuracy']:.4f}  F1-Macro={train_eval_final['f1_macro']:.4f}")
    print(f"    Val    Acc={val_eval_final['accuracy']:.4f}  F1-Macro={val_eval_final['f1_macro']:.4f}")

    # Save model and tokenizer
    fold_model_dir = output_dir / f"fold_{fold_idx}" / "model"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, fold_model_dir / "pytorch_model.pt")
    tokenizer.save_pretrained(str(fold_model_dir / "tokenizer"))
    print(f"  ✅ Model + tokenizer saved: {fold_model_dir}")

    # Build fold results
    fold_metrics = {
        "fold": fold_idx,
        "val_loss": val_eval_final["loss"],
        "val_accuracy": val_eval_final["accuracy"],
        "val_f1_macro": val_eval_final["f1_macro"],
        "val_f1_micro": val_eval_final["f1_micro"],
        "val_f1_weighted": val_eval_final["f1_weighted"],
        "train_accuracy": train_eval_final["accuracy"],
        "train_f1_macro": train_eval_final["f1_macro"],
        "best_epoch": len(epoch_log) - patience_counter,
        "total_epochs": len(epoch_log),
        "epoch_log": epoch_log,
    }
    if using_predicted_reasons:
        fold_metrics["val_rouge_l"] = float(np.mean([e.get("val_rouge_l", 1.0) for e in epoch_log]))
        fold_metrics["val_meteor"] = float(np.mean([e.get("val_meteor", 1.0) for e in epoch_log]))

    # Save fold completion marker
    completed_data = {
        "metrics": fold_metrics,
        "true_labels": val_eval_final["true"],
        "pred_labels": val_eval_final["pred"],
    }
    with open(completed_path, "w") as f:
        json.dump(completed_data, f, default=str)
    print(f"  ✅ Fold {fold_idx} completed and saved.")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  🗑️ Checkpoint deleted.")

    return fold_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Trial 5: Fine-tuned RoBERTa severity classifier (v2)"
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to df_final_nlp_bert_v2.joblib")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save final results")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory for checkpoints")
    parser.add_argument("--t5-checkpoints-dir", type=str, default="",
                        help="Directory where completed T5-base reason folds are saved")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir     = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    data_path      = resolve_path(args.data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t5_checkpoints_dir = ""
    if args.t5_checkpoints_dir:
        t5_checkpoints_dir = str(resolve_path(args.t5_checkpoints_dir))

    print(f"Loading fold data from: {data_path}")
    fold_data = joblib.load(data_path)
    num_folds = len(fold_data)
    print(f"Loaded {num_folds} folds")

    all_metrics = []
    for fold_idx, d in enumerate(fold_data):
        train_labels = [get_severity_class_id(s) for s in d["train_severity_score"]]
        val_labels   = [get_severity_class_id(s) for s in d["val_severity_score"]]

        metrics = train_one_fold(
            fold_idx     = fold_idx,
            train_X      = d["train_X"],
            train_reasons= d["train_severity_reason"],
            train_labels = train_labels,
            val_X        = d["val_X"],
            val_reasons_gt = d["val_severity_reason"],
            val_labels   = val_labels,
            device       = device,
            checkpoint_dir = checkpoint_dir,
            output_dir   = output_dir,
            t5_checkpoints_dir = t5_checkpoints_dir,
        )
        all_metrics.append(metrics)

    # Aggregate Results
    avg_metrics = {
        "task":           "severity_classification",
        "model":          MODEL_NAME,
        "max_input_len":  MAX_INPUT_LEN,
        "batch_size":     BATCH_SIZE,
        "learning_rate":  LEARNING_RATE,
        "num_epochs":     NUM_EPOCHS,
        "patience":       PATIENCE,
        "num_folds":      num_folds,
        "avg_train_acc":  float(np.mean([m["train_accuracy"] for m in all_metrics])),
        "avg_train_f1":   float(np.mean([m["train_f1_macro"] for m in all_metrics])),
        "avg_val_loss":   float(np.mean([m["val_loss"] for m in all_metrics])),
        "avg_val_acc":    float(np.mean([m["val_accuracy"] for m in all_metrics])),
        "avg_val_f1":     float(np.mean([m["val_f1_macro"] for m in all_metrics])),
        "avg_val_f1_micro": float(np.mean([m["val_f1_micro"] for m in all_metrics])),
        "avg_val_f1_weighted": float(np.mean([m["val_f1_weighted"] for m in all_metrics])),
        "per_fold_metrics": all_metrics,
    }
    if all_metrics[0].get("val_rouge_l"):
        avg_metrics["avg_val_rouge_l"] = float(np.mean([m["val_rouge_l"] for m in all_metrics]))
        avg_metrics["avg_val_meteor"] = float(np.mean([m["val_meteor"] for m in all_metrics]))

    results_path = output_dir / "results_roberta_classifier_v2.json"
    with open(results_path, "w") as f:
        json.dump(avg_metrics, f, indent=4, default=str)
    print(f"\n✅ Results saved: {results_path}")

    # Save OOF predictions
    oof_data = {"folds": []}
    for fold_idx in range(num_folds):
        comp_path = checkpoint_dir / f"fold_{fold_idx}_completed.json"
        if comp_path.exists():
            with open(comp_path, "r") as f:
                cdata = json.load(f)
            oof_data["folds"].append({
                "fold": fold_idx,
                "true_labels": cdata["true_labels"],
                "pred_labels": cdata["pred_labels"],
            })
    oof_path = output_dir / "oof_predictions_roberta_classifier_v2.json"
    with open(oof_path, "w") as f:
        json.dump(oof_data, f, default=str)
    print(f"✅ OOF predictions saved: {oof_path}")


if __name__ == "__main__":
    main()
