"""
Tier B: Fine-tuned DistilBERT regressor for severity score prediction.
Input:  complaint description text
Output: severity_score (float, 0-100 scale)
Runs on Vertex AI with GPU — 5-fold cross-validation.
Supports fold-level and epoch-level resuming from Cloud Storage checkpoints.

Usage (Vertex AI):
  python scripts/distilbert_severity_regressor.py \
    --data-path gs://bucket/path/to/data.joblib \
    --output-dir gs://bucket/path/to/output \
    --checkpoint-dir gs://bucket/path/to/checkpoints
"""
import os
import sys
import argparse
import json
import time
import random
from pathlib import Path

# Force line buffering so logs show up immediately in Google Cloud Console
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PJRT_DEVICE"] = "CUDA"

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "distilbert-base-uncased"
MAX_INPUT_LEN   = 256
BATCH_SIZE      = 32
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01
NUM_EPOCHS      = 30
SEED            = 43
PATIENCE        = 5


# ── Model ─────────────────────────────────────────────────────────────────────
class DistilBERTRegressor(nn.Module):
    """DistilBERT with a regression head for severity score prediction."""

    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # 768 → 1

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        score = self.regressor(cls_output)
        return score.squeeze(-1)  # (batch_size,)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SeverityScoreDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_len):
        self.texts     = texts
        self.scores    = scores
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = float(self.scores[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score":          torch.tensor(score, dtype=torch.float32),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Compute R², MAE, RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def resolve_path(p):
    """Support gs:// paths via /gcs/ mount on Vertex AI."""
    if p.startswith("gs://"):
        return Path(p.replace("gs://", "/gcs/", 1))
    return Path(p)


def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader, return metrics and predictions."""
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            scores    = batch["score"].to(device)

            preds = model(input_ids, attn_mask)
            loss = loss_fn(preds, scores)
            total_loss += loss.item()
            num_batches += 1

            all_true.extend(scores.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(all_true, all_pred)
    metrics["loss"] = float(avg_loss)
    return metrics, all_true, all_pred


# ── Training ─────────────────────────────────────────────────────────────────
def train_one_fold(fold_idx, train_X, train_scores,
                   val_X, val_scores,
                   device, checkpoint_dir, output_dir):
    """Train DistilBERT regressor for one fold with checkpoint resume support."""
    print(f"\n{'=' * 70}")
    print(f"  FOLD {fold_idx} — DistilBERT SEVERITY REGRESSOR")
    print(f"{'=' * 70}")

    checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_checkpoint.pt"
    completed_path  = checkpoint_dir / f"fold_{fold_idx}_completed.json"

    # ── 1. Check if fold already completed ────────────────────────────────────
    if completed_path.exists():
        print(f"✅ Fold {fold_idx} already completed. Loading from cache...")
        with open(completed_path, "r") as f:
            completed_data = json.load(f)
        return completed_data["metrics"]

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = DistilBERTRegressor(MODEL_NAME).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # ── 4. Datasets & Dataloaders ─────────────────────────────────────────────
    train_ds = SeverityScoreDataset(train_X, train_scores, tokenizer, MAX_INPUT_LEN)
    val_ds   = SeverityScoreDataset(val_X, val_scores, tokenizer, MAX_INPUT_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ── 4b. Deterministic Train subset for fast train-set evaluation ─────────
    train_X_list = list(train_X)
    train_scores_list = [float(x) for x in train_scores]
    eval_train_size = min(1000, len(train_X_list))
    rng = random.Random(SEED)
    eval_indices = rng.sample(range(len(train_X_list)), eval_train_size)
    train_X_eval = [train_X_list[i] for i in eval_indices]
    train_scores_eval = [train_scores_list[i] for i in eval_indices]
    train_eval_ds = SeverityScoreDataset(train_X_eval, train_scores_eval, tokenizer, MAX_INPUT_LEN)
    train_eval_dl = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=True)

    # ── 5. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # ── 6. Tracking ───────────────────────────────────────────────────────────
    start_epoch       = 0
    best_val_loss     = float("inf")
    best_model_state  = None
    patience_counter  = 0
    epoch_log         = []

    # ── 7. Resume from checkpoint ─────────────────────────────────────────────
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

    # ── 8. Training loop ──────────────────────────────────────────────────────
    loss_fn = nn.MSELoss()

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            scores    = batch["score"].to(device)

            preds = model(input_ids, attn_mask)
            loss = loss_fn(preds, scores)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"  Fold {fold_idx} | Epoch {epoch + 1}/{NUM_EPOCHS} | "
                      f"Step {step + 1}/{len(train_dl)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dl)
        elapsed = time.time() - t0

        # ── Evaluate on train subset ──────────────────────────────────────────
        train_metrics, _, _ = evaluate(model, train_eval_dl, device)

        # ── Evaluate on validation set ────────────────────────────────────────
        val_metrics, _, _ = evaluate(model, val_dl, device)

        # ── Log ───────────────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n  Fold {fold_idx} | Epoch {epoch + 1}/{NUM_EPOCHS} | "
              f"Time: {elapsed:.0f}s | LR: {current_lr:.2e}")
        print(f"    Train  Loss={train_metrics['loss']:.4f}  R²={train_metrics['r2']:.4f}  "
              f"MAE={train_metrics['mae']:.2f}  RMSE={train_metrics['rmse']:.2f}")
        print(f"    Val    Loss={val_metrics['loss']:.4f}  R²={val_metrics['r2']:.4f}  "
              f"MAE={val_metrics['mae']:.2f}  RMSE={val_metrics['rmse']:.2f}")

        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_r2": train_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_r2": val_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "lr": current_lr,
            "elapsed_s": elapsed,
        }
        epoch_log.append(epoch_entry)

        # ── Scheduler step ────────────────────────────────────────────────────
        scheduler.step(val_metrics["loss"])

        # ── Early stopping check ──────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✅ New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"    ⏳ No improvement ({patience_counter}/{PATIENCE})")

        # ── Save checkpoint ───────────────────────────────────────────────────
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

    # ── 9. Final evaluation with best model ──────────────────────────────────
    print(f"\n  Loading best model state for final evaluation...")
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    val_metrics_final, val_true, val_pred = evaluate(model, val_dl, device)
    train_metrics_final, _, _ = evaluate(model, train_eval_dl, device)

    print(f"\n  FOLD {fold_idx} FINAL RESULTS (best checkpoint):")
    print(f"    Train  R²={train_metrics_final['r2']:.4f}  MAE={train_metrics_final['mae']:.2f}  "
          f"RMSE={train_metrics_final['rmse']:.2f}")
    print(f"    Val    R²={val_metrics_final['r2']:.4f}  MAE={val_metrics_final['mae']:.2f}  "
          f"RMSE={val_metrics_final['rmse']:.2f}")

    # ── 10. Save model and tokenizer ──────────────────────────────────────────
    fold_model_dir = output_dir / f"fold_{fold_idx}" / "model"
    fold_model_dir.mkdir(parents=True, exist_ok=True)

    # Save the full model state
    torch.save(best_model_state, fold_model_dir / "pytorch_model.pt")
    tokenizer.save_pretrained(str(fold_model_dir / "tokenizer"))
    print(f"  ✅ Model + tokenizer saved: {fold_model_dir}")

    # ── 11. Build fold results ────────────────────────────────────────────────
    # Sample predictions
    sample_preds = []
    for i in range(min(10, len(val_true))):
        sample_preds.append({
            "input": str(val_X[i])[:200],
            "true_score": val_true[i],
            "pred_score": round(val_pred[i], 1),
        })

    fold_metrics = {
        "fold": fold_idx,
        "val_loss": val_metrics_final["loss"],
        "val_r2": val_metrics_final["r2"],
        "val_mae": val_metrics_final["mae"],
        "val_rmse": val_metrics_final["rmse"],
        "train_r2": train_metrics_final["r2"],
        "train_mae": train_metrics_final["mae"],
        "train_rmse": train_metrics_final["rmse"],
        "best_epoch": len(epoch_log) - patience_counter,
        "total_epochs": len(epoch_log),
        "epoch_log": epoch_log,
        "sample_predictions": sample_preds,
        "score_range": {
            "min": float(min(val_pred)),
            "max": float(max(val_pred)),
        },
    }

    # Save fold completion marker
    completed_data = {
        "metrics": fold_metrics,
        "true_scores": val_true,
        "pred_scores": val_pred,
    }
    with open(completed_path, "w") as f:
        json.dump(completed_data, f, default=str)
    print(f"  ✅ Fold {fold_idx} completed and saved.")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  🗑️ Checkpoint deleted (fold completed).")

    return fold_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tier B: Fine-tuned DistilBERT severity regressor"
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to df_final_nlp_bert_v2.joblib")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save final results")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory for epoch-level checkpoints")
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

    # Resolve paths
    output_dir     = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    data_path      = resolve_path(args.data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load fold data
    print(f"\nLoading fold data from: {data_path}")
    fold_data = joblib.load(data_path)
    num_folds = len(fold_data)
    print(f"Loaded {num_folds} folds")

    d0 = fold_data[0]
    print(f"  Train samples: {len(d0['train_X'])}")
    print(f"  Val samples:   {len(d0['val_X'])}")
    scores_sample = np.array(d0["train_severity_score"], dtype=float)
    print(f"  Score range:   {scores_sample.min():.1f} – {scores_sample.max():.1f}")
    print(f"  Score mean:    {scores_sample.mean():.1f}")

    # Run folds
    all_metrics = []
    for fold_idx, d in enumerate(fold_data):
        metrics = train_one_fold(
            fold_idx     = fold_idx,
            train_X      = d["train_X"],
            train_scores = d["train_severity_score"],
            val_X        = d["val_X"],
            val_scores   = d["val_severity_score"],
            device       = device,
            checkpoint_dir = checkpoint_dir,
            output_dir   = output_dir,
        )
        all_metrics.append(metrics)

    # ── Aggregate Results ─────────────────────────────────────────────────────
    avg_metrics = {
        "task":           "severity_score_regression",
        "model":          MODEL_NAME,
        "max_input_len":  MAX_INPUT_LEN,
        "batch_size":     BATCH_SIZE,
        "learning_rate":  LEARNING_RATE,
        "num_epochs":     NUM_EPOCHS,
        "patience":       PATIENCE,
        "num_folds":      num_folds,
        "avg_train_r2":   float(np.mean([m["train_r2"]   for m in all_metrics])),
        "avg_train_mae":  float(np.mean([m["train_mae"]  for m in all_metrics])),
        "avg_train_rmse": float(np.mean([m["train_rmse"] for m in all_metrics])),
        "avg_val_loss":   float(np.mean([m["val_loss"]   for m in all_metrics])),
        "avg_val_r2":     float(np.mean([m["val_r2"]     for m in all_metrics])),
        "avg_val_mae":    float(np.mean([m["val_mae"]    for m in all_metrics])),
        "avg_val_rmse":   float(np.mean([m["val_rmse"]   for m in all_metrics])),
        "per_fold_metrics": all_metrics,
    }

    # Save results
    results_path = output_dir / "results_distilbert_regressor.json"
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
                "true_scores": cdata["true_scores"],
                "pred_scores": cdata["pred_scores"],
            })
    oof_path = output_dir / "oof_predictions_distilbert_reg.json"
    with open(oof_path, "w") as f:
        json.dump(oof_data, f, default=str)
    print(f"✅ OOF predictions saved: {oof_path}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  DistilBERT SEVERITY REGRESSOR — FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Average Metrics (across {num_folds} folds):")
    print(f"    Train R²:    {avg_metrics['avg_train_r2']:.4f}")
    print(f"    Train MAE:   {avg_metrics['avg_train_mae']:.2f}")
    print(f"    Train RMSE:  {avg_metrics['avg_train_rmse']:.2f}")
    print(f"    Val Loss:    {avg_metrics['avg_val_loss']:.4f}")
    print(f"    Val R²:      {avg_metrics['avg_val_r2']:.4f}")
    print(f"    Val MAE:     {avg_metrics['avg_val_mae']:.2f}")
    print(f"    Val RMSE:    {avg_metrics['avg_val_rmse']:.2f}")

    print(f"\n  Per-Fold Breakdown:")
    print(f"  {'Fold':<6} {'Val R²':>8} {'Val MAE':>8} {'Val RMSE':>10} {'Trn R²':>8} {'Epochs':>8}")
    print(f"  {'-' * 50}")
    for m in all_metrics:
        print(f"  {m['fold']:<6} {m['val_r2']:>8.4f} {m['val_mae']:>8.2f} "
              f"{m['val_rmse']:>10.2f} {m['train_r2']:>8.4f} {m['total_epochs']:>8}")

    print(f"\n  📊 T5-small baseline: Val R² = 0.5283, MAE = 8.50, RMSE = 12.38")
    improvement = avg_metrics["avg_val_r2"] - 0.5283
    print(f"  {'📈 Improvement' if improvement > 0 else '📉 Difference'}: {improvement:+.4f} R²")

    print(f"\n  Sample Predictions (Fold 0):")
    if all_metrics[0].get("sample_predictions"):
        for sp in all_metrics[0]["sample_predictions"][:5]:
            print(f"    Input:  {sp['input'][:80]}...")
            print(f"    True:   {sp['true_score']:.1f}")
            print(f"    Pred:   {sp['pred_score']:.1f}")
            print()

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
