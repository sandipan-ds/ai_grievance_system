"""
T5-base fine-tuning for severity reason generation only.
Input:  complaint description
Output: severity reason text
Runs on Vertex AI or locally with GPU — 5-fold cross-validation.
"""
import os
import sys
import shutil
import random
import argparse
import json
import time
from pathlib import Path

# Force line buffering so logs show up immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PJRT_DEVICE"] = "CUDA"

import joblib
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
MODEL_NAME        = "t5-base"
MAX_INPUT_LEN     = 512
MAX_OUTPUT_LEN    = 128
BATCH_SIZE        = 8  # Reduced for T5-base VRAM limits
LEARNING_RATE     = 1e-4  # Standard learning rate for T5-base fine-tuning
WEIGHT_DECAY      = 0.01
NUM_EPOCHS        = 200
SEED              = 43
TASK_NAME         = "severity_reason_generation"
INPUT_PREFIX      = "predict severity reason: "


def compute_rouge_l(pred, ref):
    """Simple token-level ROUGE-L F1 (no external deps)."""
    pred_tokens = str(pred).lower().split()
    ref_tokens  = str(ref).lower().split()
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    # LCS via DP
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
    """Compute METEOR score for single hypothesis and reference."""
    pred_tokens = str(pred).lower().split()
    ref_tokens  = str(ref).lower().split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    return float(meteor_score([ref_tokens], pred_tokens))


# ── Dataset ───────────────────────────────────────────────────────────────────
class ReasonDataset(Dataset):
    def __init__(self, texts, reasons, tokenizer, max_input_len, max_output_len):
        self.texts         = texts
        self.reasons       = reasons
        self.tokenizer     = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text  = INPUT_PREFIX + str(self.texts[idx])
        target_text = str(self.reasons[idx])

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding token ids with -100 so they are ignored in the loss
        labels = target_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# ── Training ─────────────────────────────────────────────────────────────────
def train_one_fold(fold_idx, train_X, train_reasons, train_scores,
                   val_X, val_reasons, val_scores,
                   device, checkpoint_dir, output_dir):
    """Train T5-base for one fold with checkpoint resume support."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx} — SEVERITY REASON GENERATION (T5-base)")
    print(f"{'='*60}")

    checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_checkpoint.pt"
    completed_path  = checkpoint_dir / f"fold_{fold_idx}_completed.json"

    # ── 1. Check if fold already completed ────────────────────────────────────
    if completed_path.exists():
        print(f"✅ Fold {fold_idx} already completed. Loading from cache...")
        with open(completed_path, "r") as f:
            completed_data = json.load(f)
        return completed_data["metrics"]

    # ── 2. Tokenizer & Model ──────────────────────────────────────────────────
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    # ── 3. Datasets & Dataloaders ─────────────────────────────────────────────
    train_ds = ReasonDataset(train_X, train_reasons, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    val_ds   = ReasonDataset(val_X, val_reasons, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Deterministic subset for fast training progress evaluation
    train_X_list = list(train_X)
    train_reasons_list = [str(x) for x in train_reasons]
    eval_train_size = min(500, len(train_X_list))
    rng = random.Random(SEED)
    eval_indices = rng.sample(range(len(train_X_list)), eval_train_size)

    train_X_eval = [train_X_list[i] for i in eval_indices]
    train_reasons_eval = [train_reasons_list[i] for i in eval_indices]
    train_eval_ds = ReasonDataset(train_X_eval, train_reasons_eval, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    train_eval_dl = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Deterministic validation subset for fast training progress evaluation
    val_X_list = list(val_X)
    val_reasons_list = [str(x) for x in val_reasons]
    eval_val_size = min(500, len(val_X_list))
    val_rng = random.Random(SEED + 1)
    val_eval_indices = val_rng.sample(range(len(val_X_list)), eval_val_size)

    val_X_eval = [val_X_list[i] for i in val_eval_indices]
    val_reasons_eval = [val_reasons_list[i] for i in val_eval_indices]
    val_eval_ds = ReasonDataset(val_X_eval, val_reasons_eval, tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    val_eval_dl = DataLoader(val_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── 4. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # ── 5. Tracking variables ─────────────────────────────────────────────────
    start_epoch       = 0
    best_val_loss     = float("inf")
    best_model_state  = None
    patience          = 5
    patience_counter  = 0

    # ── 6. Resume from checkpoint ─────────────────────────────────────────────
    if checkpoint_path.exists():
        print(f"🔄 Checkpoint found at {checkpoint_path}. Restoring state...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch      = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_loss    = checkpoint["best_val_loss"]
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
            print(f"⚠️ Error loading checkpoint: {e}. Starting from scratch.")

    # ── 7. Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss

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

        # ── 8. Validation Loss (Fast Teacher-Forced Pass) ────────────────────────
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                val_loss_sum += outputs.loss.item()

        avg_val_loss = val_loss_sum / len(val_dl)

        # ── 9. Generate and Evaluate Predictions (Every Epoch) ────────────────
        train_preds = []
        with torch.no_grad():
            for batch in train_eval_dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=MAX_OUTPUT_LEN,
                    num_beams=1,  # greedy for speed
                    do_sample=False,
                )
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                train_preds.extend(decoded)
        
        train_rouges = [compute_rouge_l(p, t) for p, t in zip(train_preds, train_reasons_eval)]
        train_meteors = [compute_meteor(p, t) for p, t in zip(train_preds, train_reasons_eval)]
        train_rouge = float(np.mean(train_rouges))
        train_meteor = float(np.mean(train_meteors))

        # Evaluate on validation subset
        val_preds = []
        with torch.no_grad():
            for batch in val_eval_dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=MAX_OUTPUT_LEN,
                    num_beams=1,  # greedy for speed
                    do_sample=False,
                )
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                val_preds.extend(decoded)
        
        val_rouges = [compute_rouge_l(p, t) for p, t in zip(val_preds, val_reasons_eval)]
        val_meteors = [compute_meteor(p, t) for p, t in zip(val_preds, val_reasons_eval)]
        val_rouge = float(np.mean(val_rouges))
        val_meteor = float(np.mean(val_meteors))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n    Epoch {epoch+1} Summary:")
        print(f"      [Train] Loss: {avg_train_loss:.4f} | ROUGE-L: {train_rouge:.4f} | METEOR: {train_meteor:.4f}")
        print(f"      [Val]   Loss: {avg_val_loss:.4f} | ROUGE-L: {val_rouge:.4f} | METEOR: {val_meteor:.4f}")
        print(f"      [LR]    {current_lr:.6f}\n")

        # ── 10. Scheduler & Early Stopping ────────────────────────────────────
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    New best Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    patience_counter: {patience_counter}/{patience}")

        # Save checkpoint
        local_ckpt_temp = Path("/tmp") / f"t5_base_fold_{fold_idx}.pt"
        os.makedirs(os.path.dirname(local_ckpt_temp), exist_ok=True)
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss":        best_val_loss,
            "patience_counter":     patience_counter,
            "best_model_state":     best_model_state,
            "rng_state":            torch.get_rng_state(),
            "cuda_rng_state":       torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "np_rng_state":         np.random.get_state(),
            "random_rng_state":     random.getstate(),
        }, local_ckpt_temp)
        shutil.copy2(local_ckpt_temp, checkpoint_path)

        if patience_counter >= patience:
            print(f"    Early stopping triggered.")
            break

    # ── 11. Final Inference on Validation Set using Best Model ────────────────
    model.load_state_dict(best_model_state)
    model.eval()

    final_val_preds = []
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=MAX_OUTPUT_LEN,
                num_beams=1,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            final_val_preds.extend(decoded)

    val_reasons_list = [str(x) for x in val_reasons]
    val_scores_list = [float(x) for x in val_scores]
    rouges = [compute_rouge_l(p, t) for p, t in zip(final_val_preds, val_reasons_list)]
    meteors = [compute_meteor(p, t) for p, t in zip(final_val_preds, val_reasons_list)]
    final_val_rouge = float(np.mean(rouges))
    final_val_meteor = float(np.mean(meteors))

    metrics = {
        "fold":          fold_idx,
        "val_loss":      float(best_val_loss),
        "val_rouge_l":   final_val_rouge,
        "val_meteor":    final_val_meteor,
    }

    # Save model + tokenizer
    fold_model_dir = output_dir / f"fold_{fold_idx}"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_model_dir / "model")
    tokenizer.save_pretrained(fold_model_dir / "tokenizer")

    # Save fold sentinel file with predictions
    completed_data = {
        "metrics": metrics,
        "true_scores": val_scores_list,
        "true_reasons": val_reasons_list,
        "pred_reasons": final_val_preds,
    }
    local_comp_temp = Path("/tmp") / f"t5_base_completed_fold_{fold_idx}.json"
    with open(local_comp_temp, "w") as f:
        json.dump(completed_data, f, default=str)
    shutil.copy2(local_comp_temp, completed_path)
    print(f"✅ Fold {fold_idx} completed. Saved to {completed_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",       type=str, required=True)
    parser.add_argument("--output-dir",      type=str, required=True)
    parser.add_argument("--checkpoint-dir",  type=str, required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def resolve_path(p):
        if p.startswith("gs://"):
            return Path(p.replace("gs://", "/gcs/", 1))
        return Path(p)

    output_dir     = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    data_path      = resolve_path(args.data_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fold data from: {data_path}")
    fold_data = joblib.load(data_path)

    all_metrics = []

    for fold_idx, d in enumerate(fold_data):
        metrics = train_one_fold(
            fold_idx     = fold_idx,
            train_X      = d["train_X"],
            train_reasons= d["train_severity_reason"],
            train_scores = d["train_severity_score"],
            val_X        = d["val_X"],
            val_reasons  = d["val_severity_reason"],
            val_scores   = d["val_severity_score"],
            device       = device,
            checkpoint_dir = checkpoint_dir,
            output_dir   = output_dir,
        )
        all_metrics.append(metrics)

    avg_metrics = {
        "task":             TASK_NAME,
        "model":            MODEL_NAME,
        "max_input_len":    MAX_INPUT_LEN,
        "max_output_len":   MAX_OUTPUT_LEN,
        "batch_size":       BATCH_SIZE,
        "learning_rate":    LEARNING_RATE,
        "num_epochs":       NUM_EPOCHS,
        "num_folds":        len(fold_data),
        "avg_val_loss":     float(np.mean([m["val_loss"] for m in all_metrics])),
        "avg_val_rouge_l":  float(np.mean([m["val_rouge_l"] for m in all_metrics])),
        "avg_val_meteor":   float(np.mean([m["val_meteor"] for m in all_metrics])),
        "per_fold_metrics": all_metrics,
    }

    # Save overall results
    results_path = output_dir / "results_t5_base_reason.json"
    with open(results_path, "w") as f:
        json.dump(avg_metrics, f, indent=4, default=str)
    print(f"\n✅ Results saved: {results_path}")

    # Build and save OOF reasons
    oof_data = {"folds": []}
    for fold_idx in range(len(fold_data)):
        comp_path = checkpoint_dir / f"fold_{fold_idx}_completed.json"
        if comp_path.exists():
            with open(comp_path, "r") as f:
                cdata = json.load(f)
            oof_data["folds"].append({
                "fold": fold_idx,
                "true_scores":  cdata["true_scores"],
                "true_reasons": cdata["true_reasons"],
                "pred_reasons": cdata["pred_reasons"],
            })
    oof_path = output_dir / "oof_reasons.json"
    with open(oof_path, "w") as f:
        json.dump(oof_data, f, default=str)
    print(f"✅ OOF reasons saved: {oof_path}")


if __name__ == "__main__":
    main()
