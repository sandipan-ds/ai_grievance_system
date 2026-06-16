"""
Tier A: Classical regression with sentence-transformer embeddings for severity score.
Models: Ridge, Decision Tree, Random Forest, XGBoost (GPU), LightGBM
Embedding: all-MiniLM-L6-v2 (384-dim)
5-fold cross-validation using pre-built fold splits from df_final_nlp_bert_v2.joblib.

Usage (local):
  python scripts/xgb_severity_regressor.py \
    --data-path data/processed/df_final_nlp_bert_v2.joblib \
    --output-dir models/severity/dataset_v2/regression_trial/tier_a_xgb \
    --checkpoint-dir checkpoints/severity_xgb

Usage (Vertex AI — paths are resolved via /gcs/ mount):
  python scripts/xgb_severity_regressor.py \
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

import numpy as np
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384
SEED                 = 43

MODEL_CONFIGS = {
    "ridge": {
        "alpha": 1.0,
    },
    "decision_tree": {
        "max_depth": 10,
        "min_samples_leaf": 5,
        "random_state": SEED,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 3,
        "n_jobs": -1,
        "random_state": SEED,
    },
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "tree_method": "hist",
        "eval_metric": ["rmse", "mae"],
    },
    "lightgbm": {
        "n_estimators": 1000,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    },
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


def print_separator(title, char="─", width=70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# ── Embedding Extraction ─────────────────────────────────────────────────────
def extract_embeddings(all_texts, device_str="cuda"):
    """Extract sentence-transformer embeddings for all unique texts."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = device_str if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    print(f"  Encoding {len(all_texts)} texts (batch_size=256)...")
    t0 = time.time()
    embeddings = model.encode(
        all_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  ✅ Embeddings extracted: {embeddings.shape} in {elapsed:.1f}s")
    return embeddings


# ── Model Training Functions ─────────────────────────────────────────────────
def train_ridge(X_train, y_train, X_val, y_val, fold_idx):
    from sklearn.linear_model import Ridge

    print(f"\n    [Ridge] Fold {fold_idx} — Training...")
    t0 = time.time()
    model = Ridge(**MODEL_CONFIGS["ridge"])
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)

    print(f"    [Ridge] Fold {fold_idx} — Done in {elapsed:.1f}s")
    print(f"      Train  R²={train_m['r2']:.4f}  MAE={train_m['mae']:.2f}  RMSE={train_m['rmse']:.2f}")
    print(f"      Val    R²={val_m['r2']:.4f}  MAE={val_m['mae']:.2f}  RMSE={val_m['rmse']:.2f}")

    return model, train_m, val_m, val_preds.tolist(), None


def train_decision_tree(X_train, y_train, X_val, y_val, fold_idx):
    from sklearn.tree import DecisionTreeRegressor

    print(f"\n    [Decision Tree] Fold {fold_idx} — Training...")
    t0 = time.time()
    model = DecisionTreeRegressor(**MODEL_CONFIGS["decision_tree"])
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)

    print(f"    [Decision Tree] Fold {fold_idx} — Done in {elapsed:.1f}s")
    print(f"      Train  R²={train_m['r2']:.4f}  MAE={train_m['mae']:.2f}  RMSE={train_m['rmse']:.2f}")
    print(f"      Val    R²={val_m['r2']:.4f}  MAE={val_m['mae']:.2f}  RMSE={val_m['rmse']:.2f}")

    return model, train_m, val_m, val_preds.tolist(), None


def train_random_forest(X_train, y_train, X_val, y_val, fold_idx):
    from sklearn.ensemble import RandomForestRegressor

    print(f"\n    [Random Forest] Fold {fold_idx} — Training...")
    t0 = time.time()
    model = RandomForestRegressor(**MODEL_CONFIGS["random_forest"])
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)

    print(f"    [Random Forest] Fold {fold_idx} — Done in {elapsed:.1f}s")
    print(f"      Train  R²={train_m['r2']:.4f}  MAE={train_m['mae']:.2f}  RMSE={train_m['rmse']:.2f}")
    print(f"      Val    R²={val_m['r2']:.4f}  MAE={val_m['mae']:.2f}  RMSE={val_m['rmse']:.2f}")

    return model, train_m, val_m, val_preds.tolist(), None


def train_xgboost(X_train, y_train, X_val, y_val, fold_idx):
    from xgboost import XGBRegressor

    # Try GPU, fallback to CPU
    params = dict(MODEL_CONFIGS["xgboost"])
    try:
        import torch
        if torch.cuda.is_available():
            params["device"] = "cuda"
            print(f"\n    [XGBoost GPU] Fold {fold_idx} — Training with CUDA...")
        else:
            params["device"] = "cpu"
            print(f"\n    [XGBoost CPU] Fold {fold_idx} — Training on CPU (no CUDA)...")
    except ImportError:
        params["device"] = "cpu"
        print(f"\n    [XGBoost CPU] Fold {fold_idx} — Training on CPU...")

    t0 = time.time()
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10,
    )
    elapsed = time.time() - t0

    # Get training history
    eval_result = model.evals_result()

    # Best iteration
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else params["n_estimators"]

    # Final predictions using best iteration
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)

    print(f"    [XGBoost] Fold {fold_idx} — Done in {elapsed:.1f}s (best_iteration={best_iter})")
    print(f"      Train  R²={train_m['r2']:.4f}  MAE={train_m['mae']:.2f}  RMSE={train_m['rmse']:.2f}")
    print(f"      Val    R²={val_m['r2']:.4f}  MAE={val_m['mae']:.2f}  RMSE={val_m['rmse']:.2f}")

    return model, train_m, val_m, val_preds.tolist(), eval_result


def train_lightgbm(X_train, y_train, X_val, y_val, fold_idx):
    import lightgbm as lgb

    print(f"\n    [LightGBM] Fold {fold_idx} — Training...")
    t0 = time.time()
    params = dict(MODEL_CONFIGS["lightgbm"])
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric=["rmse", "mae"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=10),
        ],
    )
    elapsed = time.time() - t0

    # Get training history
    eval_result = model.evals_result_

    # Best iteration
    best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") else params["n_estimators"]

    # Final predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)

    print(f"    [LightGBM] Fold {fold_idx} — Done in {elapsed:.1f}s (best_iteration={best_iter})")
    print(f"      Train  R²={train_m['r2']:.4f}  MAE={train_m['mae']:.2f}  RMSE={train_m['rmse']:.2f}")
    print(f"      Val    R²={val_m['r2']:.4f}  MAE={val_m['mae']:.2f}  RMSE={val_m['rmse']:.2f}")

    return model, train_m, val_m, val_preds.tolist(), eval_result


# ── Model dispatcher ─────────────────────────────────────────────────────────
TRAINERS = {
    "ridge":         train_ridge,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "xgboost":       train_xgboost,
    "lightgbm":      train_lightgbm,
}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tier A: Classical severity regressors with sentence-transformer embeddings"
    )
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to df_final_nlp_bert_v2.joblib")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results, OOF predictions, models")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory for fold-level checkpoints")
    args = parser.parse_args()

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)

    # Resolve paths
    data_path      = resolve_path(args.data_path)
    output_dir     = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load fold data ─────────────────────────────────────────────────────
    print_separator("1. LOADING DATA")
    print(f"  Data path: {data_path}")
    fold_data = joblib.load(data_path)
    num_folds = len(fold_data)
    print(f"  Loaded {num_folds} folds")

    d0 = fold_data[0]
    print(f"  Train samples:  {len(d0['train_X'])}")
    print(f"  Val samples:    {len(d0['val_X'])}")
    scores_sample = np.array(d0["train_severity_score"], dtype=float)
    print(f"  Score range:    {scores_sample.min():.1f} – {scores_sample.max():.1f}")
    print(f"  Score mean:     {scores_sample.mean():.1f}")

    # ── 2. Collect all unique texts and extract embeddings ────────────────────
    print_separator("2. EXTRACTING EMBEDDINGS")

    embeddings_cache_path = checkpoint_dir / "embeddings_cache.npz"
    texts_cache_path      = checkpoint_dir / "texts_cache.json"

    if embeddings_cache_path.exists() and texts_cache_path.exists():
        print("  ✅ Loading cached embeddings...")
        cached = np.load(embeddings_cache_path)
        all_embeddings = cached["embeddings"]
        with open(texts_cache_path, "r") as f:
            all_texts_ordered = json.load(f)
        print(f"  Loaded {len(all_texts_ordered)} cached embeddings: {all_embeddings.shape}")
    else:
        # Collect all unique texts across folds (fold 0 train + val = full dataset)
        all_texts_set = []
        seen = set()
        for d in fold_data:
            for t in list(d["train_X"]) + list(d["val_X"]):
                t_str = str(t).strip()
                if t_str not in seen:
                    seen.add(t_str)
                    all_texts_set.append(t_str)

        all_texts_ordered = all_texts_set
        print(f"  Unique texts: {len(all_texts_ordered)}")

        # Extract embeddings
        all_embeddings = extract_embeddings(all_texts_ordered, device_str="cuda")

        # Cache embeddings
        np.savez_compressed(embeddings_cache_path, embeddings=all_embeddings)
        with open(texts_cache_path, "w") as f:
            json.dump(all_texts_ordered, f)
        print(f"  ✅ Embeddings cached to {embeddings_cache_path}")

    # Build text → index lookup
    text_to_idx = {t: i for i, t in enumerate(all_texts_ordered)}

    # ── 3. Train all models across all folds ──────────────────────────────────
    print_separator("3. TRAINING MODELS")

    # Results storage: {model_name: {"per_fold": [...], "oof_preds": [...]}}
    all_results = {}
    for model_name in TRAINERS:
        all_results[model_name] = {
            "per_fold_train_metrics": [],
            "per_fold_val_metrics": [],
            "oof_true_scores": [],
            "oof_pred_scores": [],
            "training_history": [],     # For XGBoost/LightGBM per-round logs
        }

    for fold_idx, d in enumerate(fold_data):
        print_separator(f"FOLD {fold_idx}/{num_folds - 1}", char="═")

        # Check if fold already completed
        fold_completed_path = checkpoint_dir / f"fold_{fold_idx}_completed.json"
        if fold_completed_path.exists():
            print(f"  ✅ Fold {fold_idx} already completed. Loading cached results...")
            with open(fold_completed_path, "r") as f:
                cached_fold = json.load(f)
            for model_name in TRAINERS:
                if model_name in cached_fold:
                    cm = cached_fold[model_name]
                    all_results[model_name]["per_fold_train_metrics"].append(cm["train_metrics"])
                    all_results[model_name]["per_fold_val_metrics"].append(cm["val_metrics"])
                    all_results[model_name]["oof_true_scores"].extend(cm["true_scores"])
                    all_results[model_name]["oof_pred_scores"].extend(cm["pred_scores"])
                    if cm.get("training_history"):
                        all_results[model_name]["training_history"].append(cm["training_history"])
            continue

        # Prepare features
        train_texts = [str(t).strip() for t in d["train_X"]]
        val_texts   = [str(t).strip() for t in d["val_X"]]
        train_scores = np.array(d["train_severity_score"], dtype=float)
        val_scores   = np.array(d["val_severity_score"], dtype=float)

        # Map texts → embeddings
        train_emb = np.array([all_embeddings[text_to_idx[t]] for t in train_texts])
        val_emb   = np.array([all_embeddings[text_to_idx[t]] for t in val_texts])

        # Add complaint length as extra feature
        train_lengths = np.array([len(t) for t in train_texts], dtype=float).reshape(-1, 1)
        val_lengths   = np.array([len(t) for t in val_texts], dtype=float).reshape(-1, 1)

        # Concatenate: embeddings (384) + length (1) = 385 features
        X_train = np.hstack([train_emb, train_lengths])
        X_val   = np.hstack([val_emb, val_lengths])

        print(f"  Features shape: Train={X_train.shape}, Val={X_val.shape}")

        # Train each model
        fold_cache = {}
        for model_name, trainer_fn in TRAINERS.items():
            model, train_m, val_m, val_preds, history = trainer_fn(
                X_train, train_scores, X_val, val_scores, fold_idx
            )

            # Store results
            all_results[model_name]["per_fold_train_metrics"].append(train_m)
            all_results[model_name]["per_fold_val_metrics"].append(val_m)
            all_results[model_name]["oof_true_scores"].extend(val_scores.tolist())
            all_results[model_name]["oof_pred_scores"].extend(val_preds)
            if history:
                all_results[model_name]["training_history"].append(history)

            # Cache for checkpoint
            fold_cache[model_name] = {
                "train_metrics": train_m,
                "val_metrics": val_m,
                "true_scores": val_scores.tolist(),
                "pred_scores": val_preds,
                "training_history": history,
            }

            # Save model for this fold
            model_save_path = output_dir / f"{model_name}_fold_{fold_idx}.joblib"
            joblib.dump(model, model_save_path)
            print(f"      Model saved: {model_save_path.name}")

        # Save fold checkpoint
        with open(fold_completed_path, "w") as f:
            json.dump(fold_cache, f, default=str)
        print(f"\n  ✅ Fold {fold_idx} checkpoint saved.")

    # ── 4. Aggregate results and produce comparative table ────────────────────
    print_separator("4. COMPARATIVE RESULTS TABLE", char="═")

    summary = {}
    for model_name in TRAINERS:
        res = all_results[model_name]
        train_metrics_list = res["per_fold_train_metrics"]
        val_metrics_list   = res["per_fold_val_metrics"]

        summary[model_name] = {
            "avg_train_r2":   float(np.mean([m["r2"]   for m in train_metrics_list])),
            "avg_train_mae":  float(np.mean([m["mae"]  for m in train_metrics_list])),
            "avg_train_rmse": float(np.mean([m["rmse"] for m in train_metrics_list])),
            "avg_val_r2":     float(np.mean([m["r2"]   for m in val_metrics_list])),
            "avg_val_mae":    float(np.mean([m["mae"]  for m in val_metrics_list])),
            "avg_val_rmse":   float(np.mean([m["rmse"] for m in val_metrics_list])),
            "std_val_r2":     float(np.std([m["r2"]    for m in val_metrics_list])),
            "per_fold_train": train_metrics_list,
            "per_fold_val":   val_metrics_list,
        }

    # Print comparative table
    header = f"  {'Model':<18} {'Train R²':>10} {'Train MAE':>10} {'Trn RMSE':>10} {'Val R²':>10} {'Val MAE':>10} {'Val RMSE':>10}"
    sep_line = f"  {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}"
    print(header)
    print(sep_line)

    best_model_name = None
    best_val_r2 = -float("inf")

    for model_name in TRAINERS:
        s = summary[model_name]
        marker = ""
        if s["avg_val_r2"] > best_val_r2:
            best_val_r2 = s["avg_val_r2"]
            best_model_name = model_name

        print(f"  {model_name:<18} {s['avg_train_r2']:>10.4f} {s['avg_train_mae']:>10.2f} "
              f"{s['avg_train_rmse']:>10.2f} {s['avg_val_r2']:>10.4f} {s['avg_val_mae']:>10.2f} "
              f"{s['avg_val_rmse']:>10.2f}")

    print(sep_line)
    print(f"\n  🏆 BEST MODEL: {best_model_name} (Val R² = {best_val_r2:.4f})")

    # T5-small baseline for comparison
    print(f"\n  📊 T5-small baseline (Trial 1): Val R² = 0.5283, MAE = 8.50, RMSE = 12.38")
    improvement = best_val_r2 - 0.5283
    print(f"  {'📈 Improvement' if improvement > 0 else '📉 Difference'}: {improvement:+.4f} R²")

    # ── 5. Save results ──────────────────────────────────────────────────────
    print_separator("5. SAVING RESULTS")

    # Full results JSON
    results = {
        "task": "severity_score_regression",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "num_folds": num_folds,
        "seed": SEED,
        "feature_dim": EMBEDDING_DIM + 1,  # embeddings + complaint_length
        "models": summary,
        "best_model": best_model_name,
        "best_val_r2": best_val_r2,
        "model_configs": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in MODEL_CONFIGS.items()},
    }

    results_path = output_dir / "results_regression_severity.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"  ✅ Results saved: {results_path}")

    # OOF predictions for ALL models
    oof_all = {}
    for model_name in TRAINERS:
        res = all_results[model_name]
        oof_all[model_name] = {
            "true_scores": res["oof_true_scores"],
            "pred_scores": res["oof_pred_scores"],
        }

    oof_path = output_dir / "oof_predictions_all_models.json"
    with open(oof_path, "w") as f:
        json.dump(oof_all, f, default=str)
    print(f"  ✅ OOF predictions (all models) saved: {oof_path}")

    # Save training histories (XGBoost/LightGBM per-round metrics)
    for model_name in ["xgboost", "lightgbm"]:
        if all_results[model_name]["training_history"]:
            hist_path = output_dir / f"training_history_{model_name}.json"
            with open(hist_path, "w") as f:
                json.dump(all_results[model_name]["training_history"], f, default=str)
            print(f"  ✅ Training history saved: {hist_path}")

    # ── 6. Final Summary ─────────────────────────────────────────────────────
    print_separator("FINAL SUMMARY", char="═")
    print(f"  Embedding Model:  {EMBEDDING_MODEL_NAME}")
    print(f"  Feature Dim:      {EMBEDDING_DIM + 1} (384 embeddings + 1 length)")
    print(f"  Folds:            {num_folds}")
    print(f"  Models Trained:   {', '.join(TRAINERS.keys())}")
    print(f"  Best Model:       {best_model_name}")
    print(f"  Best Val R²:      {best_val_r2:.4f}")
    print(f"  Best Val MAE:     {summary[best_model_name]['avg_val_mae']:.2f}")
    print(f"  Best Val RMSE:    {summary[best_model_name]['avg_val_rmse']:.2f}")
    print(f"\n  Output dir: {output_dir}")
    print(f"  Files saved:")
    for f_name in sorted(os.listdir(output_dir)):
        f_path = output_dir / f_name
        size_mb = f_path.stat().st_size / 1024 ** 2 if f_path.is_file() else 0
        print(f"    {f_name} ({size_mb:.2f} MB)" if size_mb > 0.01 else f"    {f_name}")

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    main()
