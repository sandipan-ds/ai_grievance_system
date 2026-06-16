"""
Ensemble Hard-Label Stacking for Civic Agency Classification
=============================================================
Phase 1 of the stacking pipeline. Uses only the existing OOF hard-label
predictions (no probability vectors required).

Approach:
  1. One-hot encode each model's hard prediction → 24 binary meta-features
  2. Train a Logistic Regression meta-model via nested 5-fold CV on OOF
  3. Compare per-agency F1 against individual models and majority vote

Usage:
  python scripts/ensemble_hard_label_stacking.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
N_META_FOLDS = 5  # Nested CV folds for meta-model training

BASE_DIR = Path(r"c:\Users\sandi\Desktop\ML Working Folder\ai_grievance_system")
OUTPUT_DIR = BASE_DIR / "charts_and_graphs" / "civic_agency_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_DIR = BASE_DIR / "models" / "civic_bodies" / "dataset_v2" / "ensemble_stacking"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

OOF_PATHS = {
    "DistilBERT": BASE_DIR / "models" / "civic_bodies" / "dataset_v2" / "DistilBERT" / "oof_predictions_civic.joblib",
    "RoBERTa":    BASE_DIR / "models" / "civic_bodies" / "dataset_v2" / "RoBERTa" / "oof_predictions_civic_roberta.joblib",
    "DeBERTa v3": BASE_DIR / "data" / "processed" / "oof_predictions_civic_deberta_v3.joblib",
}


def load_oof_predictions():
    """Load and validate all 3 OOF prediction files."""
    model_preds = {}
    true_labels = None
    labels = None

    for model_name, path in OOF_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"OOF file not found for {model_name}: {path}")

        data = joblib.load(path)
        model_preds[model_name] = np.array(data["pred"])

        if true_labels is None:
            true_labels = np.array(data["true"])
            labels = data["labels"]
        else:
            # Verify ground truth alignment
            assert np.array_equal(true_labels, np.array(data["true"])), \
                f"Ground truth mismatch for {model_name}!"

        print(f"  Loaded {model_name}: {len(data['pred'])} predictions from {path.name}")

    return true_labels, model_preds, labels


def one_hot_encode_predictions(preds_array, labels):
    """Convert hard label predictions to one-hot vectors."""
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n_samples = len(preds_array)
    n_classes = len(labels)

    one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i, pred in enumerate(preds_array):
        if pred in label_to_idx:
            one_hot[i, label_to_idx[pred]] = 1.0
    return one_hot


def build_meta_features(model_preds, labels):
    """Stack one-hot encoded predictions from all models into meta-feature matrix."""
    features_list = []
    feature_names = []

    for model_name, preds in model_preds.items():
        one_hot = one_hot_encode_predictions(preds, labels)
        features_list.append(one_hot)
        for label in labels:
            feature_names.append(f"{model_name}_{label}")

    X_meta = np.hstack(features_list)
    print(f"  Meta-feature matrix shape: {X_meta.shape}")
    print(f"  Features: {len(feature_names)} ({len(model_preds)} models × {len(labels)} classes)")
    return X_meta, feature_names


def majority_vote(model_preds):
    """Simple majority vote across all models."""
    model_names = list(model_preds.keys())
    n_samples = len(model_preds[model_names[0]])
    vote_preds = np.empty(n_samples, dtype=model_preds[model_names[0]].dtype)

    for i in range(n_samples):
        votes = [model_preds[name][i] for name in model_names]
        vote_preds[i] = Counter(votes).most_common(1)[0][0]

    return vote_preds


def train_meta_model_nested_cv(X_meta, y_true, labels, n_folds=N_META_FOLDS):
    """
    Train Logistic Regression meta-model via nested CV to produce OOF
    stacking predictions (avoids data leakage).
    """
    le = LabelEncoder()
    le.fit(labels)
    y_encoded = le.transform(y_true)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_stacking_preds = np.empty(len(y_true), dtype=y_true.dtype)
    oof_stacking_probs = np.zeros((len(y_true), len(labels)), dtype=np.float32)
    meta_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_encoded)):
        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train = y_encoded[train_idx]

        meta_model = LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
        )
        meta_model.fit(X_train, y_train)
        meta_models.append(meta_model)

        # Predict on validation fold
        val_pred_encoded = meta_model.predict(X_val)
        val_pred_probs = meta_model.predict_proba(X_val)
        oof_stacking_preds[val_idx] = le.inverse_transform(val_pred_encoded)
        oof_stacking_probs[val_idx] = val_pred_probs

        val_f1 = f1_score(y_encoded[val_idx], val_pred_encoded, average="macro", zero_division=0)
        print(f"    Meta-model fold {fold_idx}: val F1-Macro = {val_f1:.4f}")

    # Train final meta-model on all data for production use
    final_meta_model = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
    )
    final_meta_model.fit(X_meta, y_encoded)

    return oof_stacking_preds, oof_stacking_probs, final_meta_model, le


def evaluate_all_approaches(true_labels, model_preds, majority_preds, stacking_preds, labels):
    """Compute per-agency and overall metrics for all approaches."""
    approaches = {}

    # Individual models
    for name, preds in model_preds.items():
        approaches[name] = preds

    # Ensemble approaches
    approaches["Majority Vote"] = majority_preds
    approaches["Hard-Label Stacking (LR)"] = stacking_preds

    # Overall metrics
    overall_rows = []
    for name, preds in approaches.items():
        overall_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(true_labels, preds),
            "F1-Macro": f1_score(true_labels, preds, average="macro", zero_division=0),
            "Precision-Macro": precision_score(true_labels, preds, average="macro", zero_division=0),
            "Recall-Macro": recall_score(true_labels, preds, average="macro", zero_division=0),
        })
    df_overall = pd.DataFrame(overall_rows).set_index("Model")

    # Per-agency F1
    agency_rows = []
    for name, preds in approaches.items():
        row = {"Model": name}
        for agency in labels:
            true_binary = (true_labels == agency)
            pred_binary = (preds == agency)
            row[agency] = f1_score(true_binary, pred_binary, zero_division=0)
        agency_rows.append(row)
    df_agency_f1 = pd.DataFrame(agency_rows).set_index("Model")

    # Per-agency Precision
    agency_prec_rows = []
    for name, preds in approaches.items():
        row = {"Model": name}
        for agency in labels:
            true_binary = (true_labels == agency)
            pred_binary = (preds == agency)
            row[agency] = precision_score(true_binary, pred_binary, zero_division=0)
        agency_prec_rows.append(row)
    df_agency_prec = pd.DataFrame(agency_prec_rows).set_index("Model")

    # Per-agency Recall
    agency_rec_rows = []
    for name, preds in approaches.items():
        row = {"Model": name}
        for agency in labels:
            true_binary = (true_labels == agency)
            pred_binary = (preds == agency)
            row[agency] = recall_score(true_binary, pred_binary, zero_division=0)
        agency_rec_rows.append(row)
    df_agency_rec = pd.DataFrame(agency_rec_rows).set_index("Model")

    return df_overall, df_agency_f1, df_agency_prec, df_agency_rec, approaches


def plot_results(df_overall, df_agency_f1, labels, output_dir):
    """Generate comparison visualizations."""

    # ── Chart 1: Overall Metrics Heatmap ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    sns.heatmap(
        df_overall,
        annot=True, fmt=".4f",
        cmap="Blues", cbar=False,
        linewidths=1.0, linecolor="#eef3f7",
        annot_kws={"size": 11, "weight": "semibold"},
        ax=ax,
    )
    ax.set_title("Stacking Ensemble — Overall Metrics Comparison",
                 fontsize=13, fontweight="bold", pad=15, color="#1e3d59")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight="semibold", color="#333")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontweight="semibold", color="#333")
    ax.set_xlabel("Metric", fontsize=11, fontweight="bold", color="#1e3d59", labelpad=10)
    ax.set_ylabel("Approach", fontsize=11, fontweight="bold", color="#1e3d59", labelpad=10)
    plt.tight_layout()
    save_path = output_dir / "3.17_stacking_overall_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")

    # ── Chart 2: Per-Agency F1 Heatmap ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")

    # Custom annotation: highlight improvements
    annot_text = df_agency_f1.copy()
    for col in annot_text.columns:
        annot_text[col] = annot_text[col].apply(lambda x: f"{x*100:.1f}")

    sns.heatmap(
        df_agency_f1,
        annot=annot_text, fmt="",
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        cbar=False,
        linewidths=1.0, linecolor="#eef3f7",
        annot_kws={"size": 10, "weight": "semibold"},
        ax=ax,
    )
    ax.set_title("Per-Agency F1 Score (%) — Individual Models vs. Ensemble Approaches",
                 fontsize=13, fontweight="bold", pad=15, color="#1e3d59")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight="semibold", color="#333")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontweight="semibold", color="#333")
    ax.set_xlabel("Civic Agency", fontsize=11, fontweight="bold", color="#1e3d59", labelpad=10)
    ax.set_ylabel("Approach", fontsize=11, fontweight="bold", color="#1e3d59", labelpad=10)
    plt.tight_layout()
    save_path = output_dir / "3.17_stacking_per_agency_f1.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")

    # ── Chart 3: Delta F1 vs Best Single Model ────────────────────────────────
    # Show how much each ensemble approach improves/degrades per agency vs best single model
    individual_models = ["DistilBERT", "RoBERTa", "DeBERTa v3"]
    best_single = df_agency_f1.loc[individual_models].max(axis=0)

    ensemble_approaches = ["Majority Vote", "Hard-Label Stacking (LR)"]
    delta_rows = []
    for approach in ensemble_approaches:
        if approach in df_agency_f1.index:
            delta = df_agency_f1.loc[approach] - best_single
            delta_rows.append(delta)

    df_delta = pd.DataFrame(delta_rows, index=ensemble_approaches)

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor="white")

    # Custom annotation with +/- signs
    annot_delta = df_delta.copy()
    for col in annot_delta.columns:
        annot_delta[col] = annot_delta[col].apply(lambda x: f"{x*100:+.1f}")

    sns.heatmap(
        df_delta,
        annot=annot_delta, fmt="",
        cmap="RdYlGn", center=0,
        cbar=False,
        linewidths=1.0, linecolor="#eef3f7",
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("F1 Delta (%) vs. Best Single Model — Per Agency",
                 fontsize=13, fontweight="bold", pad=15, color="#1e3d59")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight="semibold", color="#333")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, fontweight="semibold", color="#333")
    ax.set_xlabel("Civic Agency", fontsize=11, fontweight="bold", color="#1e3d59", labelpad=10)
    ax.set_ylabel("Ensemble", fontsize=11, fontweight="bold", color="#1e3d59", labelpad=10)
    plt.tight_layout()
    save_path = output_dir / "3.17_stacking_delta_f1.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("  HARD-LABEL STACKING ENSEMBLE — CIVIC AGENCY CLASSIFICATION")
    print("=" * 70)

    # ── Step 1: Load OOF Predictions ──────────────────────────────────────────
    print("\n[Step 1] Loading OOF predictions...")
    true_labels, model_preds, labels = load_oof_predictions()
    n_samples = len(true_labels)
    print(f"  Total samples: {n_samples}, Classes: {len(labels)}")

    # ── Step 2: Build Meta-Features ───────────────────────────────────────────
    print("\n[Step 2] Building one-hot meta-features...")
    X_meta, feature_names = build_meta_features(model_preds, labels)

    # ── Step 3: Majority Vote Baseline ────────────────────────────────────────
    print("\n[Step 3] Computing majority vote baseline...")
    majority_preds = majority_vote(model_preds)
    maj_f1 = f1_score(true_labels, majority_preds, average="macro", zero_division=0)
    maj_acc = accuracy_score(true_labels, majority_preds)
    print(f"  Majority Vote — Acc: {maj_acc*100:.2f}%, F1-Macro: {maj_f1:.4f}")

    # ── Step 4: Train Meta-Model (Nested CV) ──────────────────────────────────
    print(f"\n[Step 4] Training Logistic Regression meta-model ({N_META_FOLDS}-fold nested CV)...")
    stacking_preds, stacking_probs, final_meta_model, label_encoder = \
        train_meta_model_nested_cv(X_meta, true_labels, labels)

    stacking_f1 = f1_score(true_labels, stacking_preds, average="macro", zero_division=0)
    stacking_acc = accuracy_score(true_labels, stacking_preds)
    print(f"\n  Hard-Label Stacking — Acc: {stacking_acc*100:.2f}%, F1-Macro: {stacking_f1:.4f}")

    # ── Step 5: Full Evaluation ───────────────────────────────────────────────
    print("\n[Step 5] Evaluating all approaches...")
    df_overall, df_agency_f1, df_agency_prec, df_agency_rec, all_approaches = \
        evaluate_all_approaches(true_labels, model_preds, majority_preds, stacking_preds, labels)

    print("\n  Overall Metrics:")
    print(df_overall.to_string())

    print("\n  Per-Agency F1 Scores:")
    # Format as percentages for display
    display_f1 = df_agency_f1.copy()
    for col in display_f1.columns:
        display_f1[col] = display_f1[col].apply(lambda x: f"{x*100:.1f}%")
    print(display_f1.to_string())

    # ── Step 6: Improvement Summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  IMPROVEMENT SUMMARY (vs. Best Single Model)")
    print("=" * 70)

    individual_models = ["DistilBERT", "RoBERTa", "DeBERTa v3"]
    best_single_f1 = df_overall.loc[individual_models, "F1-Macro"].max()
    best_single_name = df_overall.loc[individual_models, "F1-Macro"].idxmax()

    for approach in ["Majority Vote", "Hard-Label Stacking (LR)"]:
        if approach in df_overall.index:
            f1 = df_overall.loc[approach, "F1-Macro"]
            delta = f1 - best_single_f1
            print(f"  {approach}: F1-Macro = {f1:.4f}  ({delta:+.4f} vs {best_single_name})")

    # Per-agency delta
    best_agency_f1 = df_agency_f1.loc[individual_models].max(axis=0)
    print(f"\n  Per-Agency F1 Delta (Hard-Label Stacking vs Best Single Model):")
    for agency in labels:
        stacking_val = df_agency_f1.loc["Hard-Label Stacking (LR)", agency]
        best_val = best_agency_f1[agency]
        delta = stacking_val - best_val
        marker = "IMPROVED" if delta > 0.005 else ("DEGRADED" if delta < -0.005 else "NEUTRAL")
        print(f"    {agency:12s}: {stacking_val*100:.1f}% (delta: {delta*100:+.1f}%) [{marker}]")

    # ── Step 7: Save Results ──────────────────────────────────────────────────
    print("\n[Step 7] Saving results...")

    # Save meta-model
    meta_model_path = ENSEMBLE_DIR / "meta_model_hard_label_stacking.joblib"
    joblib.dump({
        "meta_model": final_meta_model,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "labels": labels,
        "base_model_names": list(model_preds.keys()),
    }, meta_model_path)
    print(f"  Meta-model saved: {meta_model_path}")

    # Save results JSON
    results = {
        "task": "civic_agency_stacking_ensemble",
        "phase": "hard_label_stacking",
        "n_samples": n_samples,
        "n_classes": len(labels),
        "labels": list(labels),
        "n_meta_features": X_meta.shape[1],
        "meta_model": "LogisticRegression(multinomial, balanced, C=1.0)",
        "nested_cv_folds": N_META_FOLDS,
        "overall_metrics": df_overall.to_dict(orient="index"),
        "per_agency_f1": df_agency_f1.to_dict(orient="index"),
        "per_agency_precision": df_agency_prec.to_dict(orient="index"),
        "per_agency_recall": df_agency_rec.to_dict(orient="index"),
    }
    results_path = ENSEMBLE_DIR / "results_hard_label_stacking.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"  Results JSON saved: {results_path}")

    # Save OOF stacking predictions
    oof_path = ENSEMBLE_DIR / "oof_predictions_hard_label_stacking.joblib"
    joblib.dump({
        "true": true_labels,
        "pred": stacking_preds,
        "pred_probs": stacking_probs,
        "labels": labels,
        "meta_features": X_meta,
    }, oof_path)
    print(f"  OOF stacking predictions saved: {oof_path}")

    # ── Step 8: Generate Charts ───────────────────────────────────────────────
    print("\n[Step 8] Generating comparison charts...")
    plot_results(df_overall, df_agency_f1, labels, OUTPUT_DIR)

    # ── Step 9: Confusion Matrix for Stacking ─────────────────────────────────
    cm = confusion_matrix(true_labels, stacking_preds, labels=labels)
    cm_path = ENSEMBLE_DIR / "confusion_matrix_hard_label_stacking.npy"
    np.save(cm_path, cm)
    print(f"  Confusion matrix saved: {cm_path}")

    print("\n" + "=" * 70)
    print("  PHASE 1 COMPLETE — Hard-Label Stacking Ensemble")
    print("=" * 70)
    print(f"  Results:     {results_path}")
    print(f"  Meta-model:  {meta_model_path}")
    print(f"  OOF preds:   {oof_path}")
    print(f"  Charts:      {OUTPUT_DIR / '3.17_stacking_*.png'}")


if __name__ == "__main__":
    main()
