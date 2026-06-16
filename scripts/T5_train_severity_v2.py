"""
T5-small fine-tuning for severity score + reason prediction (multi-output).
Input:  complaint description
Output: "Score: 53.8 | Reason: This is a common problem."
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
import re
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
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME        = "t5-small"
MAX_INPUT_LEN     = 256
MAX_OUTPUT_LEN    = 128
BATCH_SIZE        = 16
LEARNING_RATE     = 3e-4
WEIGHT_DECAY      = 0.01
NUM_EPOCHS        = 200
SEED              = 43
TASK_NAME         = "severity_score_reason"
INPUT_PREFIX      = "predict severity: "


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_target(score, reason):
    """Format a score + reason into the T5 target string."""
    return f"Score: {float(score):.1f} | Reason: {str(reason)}"


def parse_prediction(text):
    """
    Parse generated text like 'Score: 53.8 | Reason: common problem'.
    Returns (score_float_or_None, reason_string).
    """
    text = text.strip()
    try:
        match = re.match(r"Score:\s*([\d.]+)\s*\|\s*Reason:\s*(.*)", text, re.DOTALL)
        if match:
            score = float(match.group(1))
            reason = match.group(2).strip()
            return score, reason
    except (ValueError, AttributeError):
        pass
    # Fallback: try to extract any number
    nums = re.findall(r"[\d]+\.[\d]+|[\d]+", text)
    if nums:
        return float(nums[0]), text
    return None, text


def compute_rouge_l(pred, ref):
    """Simple token-level ROUGE-L F1 (no external deps)."""
    pred_tokens = pred.lower().split()
    ref_tokens  = ref.lower().split()
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


# ── Dataset ───────────────────────────────────────────────────────────────────
class SeverityDataset(Dataset):
    def __init__(self, texts, scores, reasons, tokenizer, max_input_len, max_output_len):
        self.texts         = texts
        self.scores        = scores
        self.reasons       = reasons
        self.tokenizer     = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text  = INPUT_PREFIX + str(self.texts[idx])
        target_text = format_target(self.scores[idx], self.reasons[idx])

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
def train_one_fold(fold_idx, train_X, train_scores, train_reasons,
                   val_X, val_scores, val_reasons,
                   device, checkpoint_dir, output_dir):
    """Train T5-small for one fold with checkpoint resume support."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx} — SEVERITY SCORE + REASON (T5-small)")
    print(f"{'='*60}")

    checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_checkpoint.pt"
    completed_path  = checkpoint_dir / f"fold_{fold_idx}_completed.json"

    # ── 1. Check if fold already completed ────────────────────────────────────
    if completed_path.exists():
        print(f"✅ Fold {fold_idx} already completed. Loading from cache...")
        with open(completed_path, "r") as f:
            completed_data = json.load(f)
        return completed_data["metrics"]

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    # ── 4. Datasets & Dataloaders ─────────────────────────────────────────────
    train_ds = SeverityDataset(train_X, train_scores, train_reasons,
                               tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    val_ds   = SeverityDataset(val_X, val_scores, val_reasons,
                               tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ── 4b. Deterministic Train subset for fast evaluation metric calculation ──
    train_X_list = list(train_X)
    train_scores_list = [float(x) for x in train_scores]
    train_reasons_list = [str(x) for x in train_reasons]

    eval_train_size = min(1000, len(train_X_list))
    # Using local Random instance with a fixed seed to be deterministic and separate from global seed
    rng = random.Random(SEED)
    eval_indices = rng.sample(range(len(train_X_list)), eval_train_size)

    train_X_eval = [train_X_list[i] for i in eval_indices]
    train_scores_eval = [train_scores_list[i] for i in eval_indices]
    train_reasons_eval = [train_reasons_list[i] for i in eval_indices]

    train_eval_ds = SeverityDataset(train_X_eval, train_scores_eval, train_reasons_eval,
                                    tokenizer, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    train_eval_dl = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True)

    # ── 5. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    # ReduceLR on plateau — monitors val_loss (lower is better)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # ── 6. Tracking variables ─────────────────────────────────────────────────
    start_epoch       = 0
    best_val_loss     = float("inf")
    best_model_state  = None
    patience          = 5
    patience_counter  = 0

    # ── 7. Resume from checkpoint ─────────────────────────────────────────────
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

    # ── 8. Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                            labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                elapsed = time.time() - t0
                print(f"    Epoch {epoch+1}/{NUM_EPOCHS} | Step {step+1}/{len(train_dl)} | "
                      f"Loss: {avg_loss:.4f} | {elapsed:.0f}s")

        avg_train_loss = total_loss / len(train_dl)

        # ── 9. Validation — loss (fast, teacher-forced forward pass) ──────────
        model.eval()
        val_loss_sum = 0

        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                                labels=labels)
                val_loss_sum += outputs.loss.item()

        avg_val_loss = val_loss_sum / len(val_dl)

        # ── 10. Validation & Train — generate predictions for MAE/ROUGE (every 3 epochs) ─
        val_mae, val_rouge = None, None
        train_mae, train_rouge = None, None

        if (epoch + 1) % 3 == 0 or epoch == start_epoch:
            # 10a. Generate predictions on training evaluation subset
            train_pred_scores, train_true_scores = [], []
            train_pred_reasons, train_true_reasons = [], []

            with torch.no_grad():
                for batch_idx, batch in enumerate(train_eval_dl):
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)

                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=MAX_OUTPUT_LEN,
                        num_beams=1,        # greedy for speed
                        do_sample=False,
                    )

                    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    start_i = batch_idx * BATCH_SIZE
                    for j, pred_text in enumerate(decoded):
                        idx = start_i + j
                        if idx >= len(train_scores_eval):
                            break
                        pred_score, pred_reason = parse_prediction(pred_text)
                        true_score = float(train_scores_eval[idx])
                        true_reason = str(train_reasons_eval[idx])

                        if pred_score is not None:
                            train_pred_scores.append(pred_score)
                            train_true_scores.append(true_score)

                        train_pred_reasons.append(pred_reason)
                        train_true_reasons.append(true_reason)

            if train_pred_scores:
                true_arr = np.array(train_true_scores)
                pred_arr = np.array(train_pred_scores)
                errors = np.abs(pred_arr - true_arr)
                train_mae = float(np.mean(errors))
                train_rmse = float(np.sqrt(np.mean(errors**2)))
                ss_res = np.sum((true_arr - pred_arr) ** 2)
                ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
                train_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                train_parseable_pct = 100.0 * len(train_pred_scores) / len(train_scores_eval)
            else:
                train_mae, train_rmse, train_r2, train_parseable_pct = 99.0, 99.0, 0.0, 0.0

            if train_pred_reasons:
                rouges = [compute_rouge_l(p, t) for p, t in
                          zip(train_pred_reasons, train_true_reasons)]
                train_rouge = float(np.mean(rouges))
            else:
                train_rouge = 0.0

            # 10b. Generate predictions on validation set
            pred_scores_list, true_scores_list = [], []
            pred_reasons_list, true_reasons_list = [], []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dl):
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)

                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=MAX_OUTPUT_LEN,
                        num_beams=1,        # greedy for speed
                        do_sample=False,
                    )

                    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    # Get true values for this batch
                    start_i = batch_idx * BATCH_SIZE
                    for j, pred_text in enumerate(decoded):
                        idx = start_i + j
                        if idx >= len(val_scores):
                            break
                        pred_score, pred_reason = parse_prediction(pred_text)
                        true_score = float(val_scores[idx])
                        true_reason = str(val_reasons[idx])

                        if pred_score is not None:
                            pred_scores_list.append(pred_score)
                            true_scores_list.append(true_score)

                        pred_reasons_list.append(pred_reason)
                        true_reasons_list.append(true_reason)

            if pred_scores_list:
                true_arr = np.array(true_scores_list)
                pred_arr = np.array(pred_scores_list)
                errors = np.abs(pred_arr - true_arr)
                val_mae = float(np.mean(errors))
                val_rmse = float(np.sqrt(np.mean(errors**2)))
                ss_res = np.sum((true_arr - pred_arr) ** 2)
                ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
                val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                parseable_pct = 100.0 * len(pred_scores_list) / len(val_scores)
            else:
                val_mae, val_rmse, val_r2, parseable_pct = 99.0, 99.0, 0.0, 0.0

            if pred_reasons_list:
                rouges = [compute_rouge_l(p, t) for p, t in
                          zip(pred_reasons_list, true_reasons_list)]
                val_rouge = float(np.mean(rouges))
            else:
                val_rouge = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n    Epoch {epoch+1} Summary:")
        print(f"      [Train] Loss: {avg_train_loss:.4f}")
        if train_mae is not None:
            print(f"      [Train] MAE: {train_mae:.2f} | RMSE: {train_rmse:.2f} | R2: {train_r2:.4f} | "
                  f"ROUGE-L: {train_rouge:.4f} | Parseable: {train_parseable_pct:.1f}%")
        print(f"      [Val]   Loss: {avg_val_loss:.4f}")
        if val_mae is not None:
            print(f"      [Val]   MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | R2: {val_r2:.4f} | "
                  f"ROUGE-L: {val_rouge:.4f} | Parseable: {parseable_pct:.1f}%")
        print(f"      [LR]    {current_lr:.6f}\n")

        # ── 11. LR scheduler step ────────────────────────────────────────────
        scheduler.step(avg_val_loss)

        # ── 12. Best model / early stopping ──────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    ✅ New best Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    patience_counter: {patience_counter}/{patience}")

        # ── 13. Save checkpoint ──────────────────────────────────────────────
        print(f"    Saving epoch checkpoint...")
        local_ckpt_temp = Path("/tmp") / f"checkpoint_fold_{fold_idx}.pt"
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
        print(f"    Checkpoint saved to {checkpoint_path}")

        if patience_counter >= patience:
            print(f"    Early stopping triggered after {epoch+1} epochs.")
            break

    # ── 14. Final evaluation with best model ─────────────────────────────────
    model.load_state_dict(best_model_state)
    model.eval()

    pred_scores_final, true_scores_final = [], []
    pred_reasons_final, true_reasons_final = [], []
    raw_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dl):
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

            start_i = batch_idx * BATCH_SIZE
            for j, pred_text in enumerate(decoded):
                idx = start_i + j
                if idx >= len(val_scores):
                    break
                pred_score, pred_reason = parse_prediction(pred_text)
                true_score = float(val_scores[idx])
                true_reason = str(val_reasons[idx])

                pred_scores_final.append(pred_score if pred_score is not None else 50.0)
                true_scores_final.append(true_score)
                pred_reasons_final.append(pred_reason)
                true_reasons_final.append(true_reason)

                # Store first 10 samples per fold for inspection
                if len(raw_predictions) < 10:
                    raw_predictions.append({
                        "input":        str(val_X[idx])[:200],
                        "true_score":   true_score,
                        "pred_score":   pred_score,
                        "true_reason":  true_reason,
                        "pred_reason":  pred_reason,
                        "raw_output":   pred_text,
                    })

    # Compute final metrics
    true_arr = np.array(true_scores_final)
    pred_arr = np.array(pred_scores_final)
    errors   = np.abs(pred_arr - true_arr)

    parseable_count = sum(1 for s in pred_scores_final if s != 50.0 or True)

    rouges = [compute_rouge_l(p, t) for p, t in
              zip(pred_reasons_final, true_reasons_final)]

    # R² score
    ss_res = np.sum((true_arr - pred_arr) ** 2)
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "fold":          fold_idx,
        "val_loss":      float(best_val_loss),
        "val_mae":       float(np.mean(errors)),
        "val_rmse":      float(np.sqrt(np.mean(errors ** 2))),
        "val_r2":        float(r2),
        "val_rouge_l":   float(np.mean(rouges)),
        "score_range":   {"min": float(np.min(pred_arr)), "max": float(np.max(pred_arr))},
        "sample_predictions": raw_predictions,
    }

    print(f"\n  Fold {fold_idx} Final Metrics:")
    print(f"    MAE:     {metrics['val_mae']:.2f}")
    print(f"    RMSE:    {metrics['val_rmse']:.2f}")
    print(f"    R²:      {metrics['val_r2']:.4f}")
    print(f"    ROUGE-L: {metrics['val_rouge_l']:.4f}")
    print(f"\n  Sample Predictions:")
    for sp in raw_predictions[:5]:
        print(f"    Input:   {sp['input'][:80]}...")
        print(f"    True:    Score={sp['true_score']:.1f} | Reason={sp['true_reason']}")
        print(f"    Pred:    Score={sp['pred_score']} | Reason={sp['pred_reason']}")
        print(f"    Raw:     {sp['raw_output']}")
        print()

    # Save model + tokenizer for this fold
    fold_model_dir = output_dir / f"fold_{fold_idx}"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_model_dir / "model")
    tokenizer.save_pretrained(fold_model_dir / "tokenizer")

    # Write completed fold sentinel
    completed_data = {
        "metrics":       metrics,
        "true_scores":   true_scores_final,
        "pred_scores":   pred_scores_final,
        "true_reasons":  true_reasons_final,
        "pred_reasons":  pred_reasons_final,
    }
    local_comp_temp = Path("/tmp") / f"completed_fold_{fold_idx}.json"
    with open(local_comp_temp, "w") as f:
        json.dump(completed_data, f, default=str)
    shutil.copy2(local_comp_temp, completed_path)
    print(f"✅ Fold {fold_idx} completed and logged to {completed_path}")

    # Remove epoch checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",       type=str, required=True,
                        help="Path to df_final_nlp_bert_v2.joblib")
    parser.add_argument("--output-dir",      type=str, required=True,
                        help="Directory to save final results")
    parser.add_argument("--checkpoint-dir",  type=str, required=True,
                        help="Directory to save checkpoints for resume")
    args = parser.parse_args()

    # Seed everything
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

    # Resolve paths (supporting gs:// via /gcs/ mount)
    def resolve_path(p):
        if p.startswith("gs://"):
            return Path(p.replace("gs://", "/gcs/", 1))
        return Path(p)

    output_dir     = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    data_path      = resolve_path(args.data_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load fold data
    print(f"\nLoading fold data from: {data_path}")
    fold_data = joblib.load(data_path)
    print(f"Loaded {len(fold_data)} folds")

    # Quick data summary
    d0 = fold_data[0]
    print(f"  Train samples: {len(d0['train_X'])}")
    print(f"  Val samples:   {len(d0['val_X'])}")
    scores_sample = np.array(d0["train_severity_score"], dtype=float)
    print(f"  Score range:   {scores_sample.min():.1f} – {scores_sample.max():.1f}")
    print(f"  Score mean:    {scores_sample.mean():.1f}")
    print(f"  Reason sample: {str(d0['train_severity_reason'][0])[:100]}")

    # Run folds
    all_metrics = []

    for fold_idx, d in enumerate(fold_data):
        metrics = train_one_fold(
            fold_idx     = fold_idx,
            train_X      = d["train_X"],
            train_scores = d["train_severity_score"],
            train_reasons= d["train_severity_reason"],
            val_X        = d["val_X"],
            val_scores   = d["val_severity_score"],
            val_reasons  = d["val_severity_reason"],
            device       = device,
            checkpoint_dir = checkpoint_dir,
            output_dir   = output_dir,
        )
        all_metrics.append(metrics)

    # ── Aggregate Results ─────────────────────────────────────────────────────
    avg_metrics = {
        "task":             TASK_NAME,
        "model":            MODEL_NAME,
        "max_input_len":    MAX_INPUT_LEN,
        "max_output_len":   MAX_OUTPUT_LEN,
        "batch_size":       BATCH_SIZE,
        "learning_rate":    LEARNING_RATE,
        "num_epochs":       NUM_EPOCHS,
        "num_folds":        len(fold_data),
        "avg_val_loss":     float(np.mean([m["val_loss"]    for m in all_metrics])),
        "avg_val_mae":      float(np.mean([m["val_mae"]     for m in all_metrics])),
        "avg_val_rmse":     float(np.mean([m["val_rmse"]    for m in all_metrics])),
        "avg_val_r2":       float(np.mean([m["val_r2"]      for m in all_metrics])),
        "avg_val_rouge_l":  float(np.mean([m["val_rouge_l"] for m in all_metrics])),
        "per_fold_metrics": all_metrics,
    }

    # Save results
    results_path = output_dir / "results_t5_severity_v2.json"
    with open(results_path, "w") as f:
        json.dump(avg_metrics, f, indent=4, default=str)
    print(f"\n✅ Results saved: {results_path}")

    # Save OOF predictions (load from completed fold files)
    oof_data = {"folds": []}
    for fold_idx in range(len(fold_data)):
        comp_path = checkpoint_dir / f"fold_{fold_idx}_completed.json"
        if comp_path.exists():
            with open(comp_path, "r") as f:
                cdata = json.load(f)
            oof_data["folds"].append({
                "fold": fold_idx,
                "true_scores":  cdata["true_scores"],
                "pred_scores":  cdata["pred_scores"],
                "true_reasons": cdata["true_reasons"],
                "pred_reasons": cdata["pred_reasons"],
            })
    oof_path = output_dir / "oof_predictions_severity_v2.json"
    with open(oof_path, "w") as f:
        json.dump(oof_data, f, default=str)
    print(f"✅ OOF predictions saved: {oof_path}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  T5-SMALL SEVERITY — FINAL TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"  Average Validation Metrics (across {len(fold_data)} folds):")
    print(f"    Loss:    {avg_metrics['avg_val_loss']:.4f}")
    print(f"    MAE:     {avg_metrics['avg_val_mae']:.2f}")
    print(f"    RMSE:    {avg_metrics['avg_val_rmse']:.2f}")
    print(f"    R²:      {avg_metrics['avg_val_r2']:.4f}")
    print(f"    ROUGE-L: {avg_metrics['avg_val_rouge_l']:.4f}")

    print(f"\n  Per-Fold Breakdown:")
    print(f"  {'Fold':<6} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'ROUGE-L':>9}")
    print(f"  {'-'*42}")
    for m in all_metrics:
        print(f"  {m['fold']:<6} {m['val_mae']:>8.2f} {m['val_rmse']:>8.2f} "
              f"{m['val_r2']:>8.4f} {m['val_rouge_l']:>9.4f}")

    print(f"\n  Sample Predictions (Fold 0):")
    if all_metrics[0].get("sample_predictions"):
        for sp in all_metrics[0]["sample_predictions"][:5]:
            print(f"    Input:  {sp['input'][:80]}...")
            print(f"    True:   Score={sp['true_score']:.1f} | {sp['true_reason']}")
            print(f"    Pred:   Score={sp['pred_score']} | {sp['pred_reason']}")
            print()

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
