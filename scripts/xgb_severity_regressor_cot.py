"""
Tier A (COT): Classical regression with sentence-transformer embeddings of complaint text + severity reason.
Models: Ridge, Decision Tree, Random Forest, XGBoost (GPU), LightGBM
Embedding: all-MiniLM-L6-v2 (384-dim)
"""
import os
import sys
import argparse
import json
import time
import random
from pathlib import Path

# Force line buffering so logs show up immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

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


def get_severity(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Medium"
    if score >= 90:
        return "Critical"
    elif score >= 80:
        return "High"
    elif score >= 50:
        return "Medium"
    elif score >= 1:
        return "Low"
    else:
        return "Non-Grievance"


def evaluate_classification(y_true_scores, y_pred_scores):
    y_true_cats = [get_severity(s) for s in y_true_scores]
    y_pred_cats = [get_severity(s) for s in y_pred_scores]
    labels_order = ['Non-Grievance', 'Low', 'Medium', 'High', 'Critical']
    
    report = classification_report(y_true_cats, y_pred_cats, labels=labels_order, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true_cats, y_pred_cats)
    prec = precision_score(y_true_cats, y_pred_cats, average='macro', zero_division=0)
    rec = recall_score(y_true_cats, y_pred_cats, average='macro', zero_division=0)
    f1 = f1_score(y_true_cats, y_pred_cats, average='macro', zero_division=0)
    
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "report_dict": report
    }


def resolve_path(p):
    if p.startswith("gs://"):
        return Path(p.replace("gs://", "/gcs/", 1))
    return Path(p)


def print_separator(title, char="─", width=70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def extract_embeddings(all_texts, device_str="cuda"):
    from sentence_transformers import SentenceTransformer
    import torch

    device = device_str if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    print(f"  Encoding {len(all_texts)} texts...")
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
    model = Ridge(**MODEL_CONFIGS["ridge"])
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)
    return model, train_m, val_m, val_preds.tolist(), None


def train_decision_tree(X_train, y_train, X_val, y_val, fold_idx):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(**MODEL_CONFIGS["decision_tree"])
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)
    return model, train_m, val_m, val_preds.tolist(), None


def train_random_forest(X_train, y_train, X_val, y_val, fold_idx):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(**MODEL_CONFIGS["random_forest"])
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)
    return model, train_m, val_m, val_preds.tolist(), None


def train_xgboost(X_train, y_train, X_val, y_val, fold_idx):
    from xgboost import XGBRegressor
    params = dict(MODEL_CONFIGS["xgboost"])
    try:
        import torch
        if torch.cuda.is_available():
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"
    except ImportError:
        params["device"] = "cpu"

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    eval_result = model.evals_result()
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)
    return model, train_m, val_m, val_preds.tolist(), eval_result


def train_lightgbm(X_train, y_train, X_val, y_val, fold_idx):
    import lightgbm as lgb
    params = dict(MODEL_CONFIGS["lightgbm"])
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric=["rmse", "mae"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
        ],
    )
    eval_result = model.evals_result_
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)
    return model, train_m, val_m, val_preds.tolist(), eval_result


TRAINERS = {
    "ridge":         train_ridge,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "xgboost":       train_xgboost,
    "lightgbm":      train_lightgbm,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--oof-reasons-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    data_path      = resolve_path(args.data_path)
    oof_path       = resolve_path(args.oof_reasons_path)
    output_dir     = resolve_path(args.output_dir)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print_separator("LOADING DATA")
    print(f"Data: {data_path}")
    print(f"OOF Reasons: {oof_path}")

    fold_data = joblib.load(data_path)
    num_folds = len(fold_data)

    with open(oof_path, "r") as f:
        oof_reasons_data = json.load(f)
    
    oof_folds_dict = {f["fold"]: f for f in oof_reasons_data["folds"]}

    all_results = {}
    for model_name in TRAINERS:
        all_results[model_name] = {
            "per_fold_train_metrics": [],
            "per_fold_val_metrics": [],
            "oof_true_scores": [],
            "oof_pred_scores": [],
            "training_history": [],
        }

    for fold_idx, d in enumerate(fold_data):
        print_separator(f"FOLD {fold_idx}/{num_folds - 1}", char="═")

        fold_completed_path = checkpoint_dir / f"fold_{fold_idx}_completed.json"
        if fold_completed_path.exists():
            print(f"  ✅ Fold {fold_idx} already completed. Loading...")
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

        train_texts = [str(t).strip() for t in d["train_X"]]
        train_reasons = [str(r).strip() for r in d["train_severity_reason"]]
        train_scores = np.array(d["train_severity_score"], dtype=float)

        val_texts = [str(t).strip() for t in d["val_X"]]
        val_reasons_pred = [str(r).strip() for r in oof_folds_dict[fold_idx]["pred_reasons"]]
        val_scores = np.array(d["val_severity_score"], dtype=float)

        assert len(val_reasons_pred) == len(val_texts), f"Alignment issue: {len(val_reasons_pred)} vs {len(val_texts)}"

        # Concatenate descriptions with reasons
        train_combined = [f"{t} | Reason: {r}" for t, r in zip(train_texts, train_reasons)]
        val_combined = [f"{t} | Reason: {r}" for t, r in zip(val_texts, val_reasons_pred)]

        # Extract embeddings
        train_emb = extract_embeddings(train_combined, device_str="cuda")
        val_emb = extract_embeddings(val_combined, device_str="cuda")

        # Append complaint length
        train_lengths = np.array([len(t) for t in train_texts], dtype=float).reshape(-1, 1)
        val_lengths = np.array([len(t) for t in val_texts], dtype=float).reshape(-1, 1)

        X_train = np.hstack([train_emb, train_lengths])
        X_val = np.hstack([val_emb, val_lengths])

        fold_cache = {}
        for model_name, trainer_fn in TRAINERS.items():
            model, train_m, val_m, val_preds, history = trainer_fn(
                X_train, train_scores, X_val, val_scores, fold_idx
            )
            all_results[model_name]["per_fold_train_metrics"].append(train_m)
            all_results[model_name]["per_fold_val_metrics"].append(val_m)
            all_results[model_name]["oof_true_scores"].extend(val_scores.tolist())
            all_results[model_name]["oof_pred_scores"].extend(val_preds)
            if history:
                all_results[model_name]["training_history"].append(history)

            fold_cache[model_name] = {
                "train_metrics": train_m,
                "val_metrics": val_m,
                "true_scores": val_scores.tolist(),
                "pred_scores": val_preds,
                "training_history": history,
            }

            # Save model
            model_save_path = output_dir / f"{model_name}_fold_{fold_idx}.joblib"
            joblib.dump(model, model_save_path)

        with open(fold_completed_path, "w") as f:
            json.dump(fold_cache, f, default=str)
        print(f"  ✅ Fold {fold_idx} completed.")

    print_separator("AGGREGATING RESULTS")
    summary = {}
    classification_summaries = {}

    for model_name in TRAINERS:
        res = all_results[model_name]
        train_metrics_list = res["per_fold_train_metrics"]
        val_metrics_list   = res["per_fold_val_metrics"]

        # Regression summary
        avg_train_r2 = float(np.mean([m["r2"] for m in train_metrics_list]))
        avg_val_r2 = float(np.mean([m["r2"] for m in val_metrics_list]))
        avg_train_mae = float(np.mean([m["mae"] for m in train_metrics_list]))
        avg_val_mae = float(np.mean([m["mae"] for m in val_metrics_list]))
        avg_train_rmse = float(np.mean([m["rmse"] for m in train_metrics_list]))
        avg_val_rmse = float(np.mean([m["rmse"] for m in val_metrics_list]))
        std_val_r2 = float(np.std([m["r2"] for m in val_metrics_list]))

        # Classification summary (Out-Of-Fold binned)
        class_metrics = evaluate_classification(res["oof_true_scores"], res["oof_pred_scores"])

        summary[model_name] = {
            "avg_train_r2": avg_train_r2,
            "avg_val_r2": avg_val_r2,
            "avg_train_mae": avg_train_mae,
            "avg_val_mae": avg_val_mae,
            "avg_train_rmse": avg_train_rmse,
            "avg_val_rmse": avg_val_rmse,
            "std_val_r2": std_val_r2,
            "accuracy": class_metrics["accuracy"],
            "precision_macro": class_metrics["precision_macro"],
            "recall_macro": class_metrics["recall_macro"],
            "f1_macro": class_metrics["f1_macro"],
            "per_fold_train": train_metrics_list,
            "per_fold_val": val_metrics_list,
        }
        classification_summaries[model_name] = class_metrics["report_dict"]

    # Print Comparative Table
    header = f"  {'Model':<18} {'Val R²':>10} {'Val MAE':>10} {'Val RMSE':>10} {'Acc':>8} {'F1-Macro':>10}"
    sep_line = f"  {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10}"
    print(header)
    print(sep_line)
    for model_name in TRAINERS:
        s = summary[model_name]
        print(f"  {model_name:<18} {s['avg_val_r2']:>10.4f} {s['avg_val_mae']:>10.2f} "
              f"{s['avg_val_rmse']:>10.2f} {s['accuracy']:>8.4f} {s['f1_macro']:>10.4f}")
    print(sep_line)

    results = {
        "task": "severity_score_cot_regression",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "num_folds": num_folds,
        "seed": SEED,
        "models": summary,
        "classification_reports": classification_summaries
    }

    results_path = output_dir / "results_cot_regression.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"✅ Results saved to {results_path}")

    # Save OOF predictions
    oof_all = {}
    for model_name in TRAINERS:
        res = all_results[model_name]
        oof_all[model_name] = {
            "true_scores": res["oof_true_scores"],
            "pred_scores": res["oof_pred_scores"],
        }
    oof_path = output_dir / "oof_predictions_cot_regression.json"
    with open(oof_path, "w") as f:
        json.dump(oof_all, f, default=str)
    print(f"✅ OOF predictions saved to {oof_path}")


if __name__ == "__main__":
    main()
