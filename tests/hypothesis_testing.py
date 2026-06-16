"""
Hypothesis Testing: Weighted Soft Voting vs RoBERTa
===================================================
Compares the civic agency out-of-fold predictions of:
1. Weighted Soft Voting Ensemble (DistilBERT=0.45, RoBERTa=0.25, DeBERTa v3=0.30)
2. RoBERTa Classifier (single model)

Performs:
- McNemar's Test (non-parametric test for paired nominal data)
- Wilcoxon Signed-Rank Test (paired comparison of fold-level macro F1 scores)
- Pairwise t-test on fold-level metrics
- Save comparison metrics and p-values to results and generate comparison charts.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(r"c:\Users\sandi\Desktop\ML Working Folder\ai_grievance_system")
OOF_PATHS = {
    "DistilBERT": BASE_DIR / "data" / "processed" / "oof_predictions_civic_distilbert_probs.joblib",
    "RoBERTa":    BASE_DIR / "data" / "processed" / "oof_predictions_civic_roberta_probs.joblib",
    "DeBERTa v3": BASE_DIR / "data" / "processed" / "oof_predictions_civic_deberta_v3_probs.joblib",
}
OUTPUT_DIR = BASE_DIR / "charts_and_graphs" / "civic_agency_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    data_dict = {}
    for name, path in OOF_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} OOF file: {path}")
        data_dict[name] = joblib.load(path)
    return data_dict

def main():
    print("Loading OOF probability predictions...")
    data = load_data()
    
    true_labels = np.array(data["RoBERTa"]["true"])
    labels = data["RoBERTa"]["labels"]
    
    le = LabelEncoder()
    le.fit(labels)
    y_true_encoded = le.transform(true_labels)
    
    # Base RoBERTa predictions
    roberta_preds = np.array(data["RoBERTa"]["pred"])
    roberta_probs = np.array(data["RoBERTa"]["pred_probs"])
    roberta_preds_encoded = le.transform(roberta_preds)
    
    # Compute Weighted Soft Voting predictions
    # Weights: DistilBERT=0.45, RoBERTa=0.25, DeBERTa v3=0.30
    w_distil = 0.45
    w_roberta = 0.25
    w_deberta = 0.30
    
    distil_probs = np.array(data["DistilBERT"]["pred_probs"])
    deberta_probs = np.array(data["DeBERTa v3"]["pred_probs"])
    
    wsv_probs = w_distil * distil_probs + w_roberta * roberta_probs + w_deberta * deberta_probs
    wsv_preds_encoded = wsv_probs.argmax(axis=1)
    wsv_preds = le.inverse_transform(wsv_preds_encoded)
    
    print(f"Total samples: {len(true_labels)}")
    print(f"RoBERTa Accuracy: {accuracy_score(true_labels, roberta_preds):.4%}")
    print(f"WSV Ensemble Accuracy: {accuracy_score(true_labels, wsv_preds):.4%}")
    
    # --- McNemar's Test ---
    # Construct 2x2 contingency table:
    #                 RoBERTa Correct    RoBERTa Incorrect
    # WSV Correct           a                  b
    # WSV Incorrect         c                  d
    
    wsv_correct = (wsv_preds_encoded == y_true_encoded)
    roberta_correct = (roberta_preds_encoded == y_true_encoded)
    
    a = np.sum(wsv_correct & roberta_correct)
    b = np.sum(wsv_correct & ~roberta_correct)
    c = np.sum(~wsv_correct & roberta_correct)
    d = np.sum(~wsv_correct & ~roberta_correct)
    
    contingency_table = np.array([[a, b], [c, d]])
    print("\nContingency Table:")
    print(f"WSV Correct & RoBERTa Correct: {a}")
    print(f"WSV Correct & RoBERTa Incorrect: {b} (WSV wins)")
    print(f"WSV Incorrect & RoBERTa Correct: {c} (RoBERTa wins)")
    print(f"WSV Incorrect & RoBERTa Incorrect: {d}")
    
    # Run McNemar test manually
    # chi2 = (|b - c| - 1)^2 / (b + c)
    stat = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value_mcnemar = chi2.sf(stat, df=1)
    
    print(f"\nMcNemar test statistic: {stat:.4f}")
    print(f"McNemar p-value: {p_value_mcnemar:.4e}")
    if p_value_mcnemar < 0.05:
        print("Conclusion: The difference in performance is statistically SIGNIFICANT (p < 0.05).")
    else:
        print("Conclusion: The difference in performance is NOT statistically significant (p >= 0.05).")
        
    # --- Fold-wise paired t-test and Wilcoxon Signed-Rank Test ---
    # To compute fold-wise metrics, we need to partition the OOF predictions back into the 5 folds.
    # In df_final_nlp_bert_v2.joblib, we have the fold splits. Let's load the dataset to retrieve fold indices.
    data_path = BASE_DIR / "data" / "processed" / "df_final_nlp_bert_v2.joblib"
    if data_path.exists():
        folds_data = joblib.load(data_path)
        roberta_fold_f1s = []
        wsv_fold_f1s = []
        roberta_fold_accs = []
        wsv_fold_accs = []
        
        start_idx = 0
        for fold_idx, fold_df in enumerate(folds_data):
            val_len = len(fold_df["val_y"])
            end_idx = start_idx + val_len
            
            y_true_fold = y_true_encoded[start_idx:end_idx]
            rob_pred_fold = roberta_preds_encoded[start_idx:end_idx]
            wsv_pred_fold = wsv_preds_encoded[start_idx:end_idx]
            
            # F1-Macro
            roberta_fold_f1s.append(f1_score(y_true_fold, rob_pred_fold, average="macro", zero_division=0))
            wsv_fold_f1s.append(f1_score(y_true_fold, wsv_pred_fold, average="macro", zero_division=0))
            
            # Accuracy
            roberta_fold_accs.append(accuracy_score(y_true_fold, rob_pred_fold))
            wsv_fold_accs.append(accuracy_score(y_true_fold, wsv_pred_fold))
            
            start_idx = end_idx
            
        print("\nFold-wise Macro F1 comparison:")
        for idx in range(5):
            print(f"  Fold {idx}: RoBERTa = {roberta_fold_f1s[idx]:.4f}, WSV = {wsv_fold_f1s[idx]:.4f} (diff = {wsv_fold_f1s[idx] - roberta_fold_f1s[idx]:+.4f})")
            
        t_stat_f1, p_val_f1 = ttest_rel(wsv_fold_f1s, roberta_fold_f1s)
        wilc_stat_f1, wilc_p_val_f1 = wilcoxon(wsv_fold_f1s, roberta_fold_f1s)
        print(f"\nPaired t-test on F1-Macro: statistic = {t_stat_f1:.4f}, p-value = {p_val_f1:.4f}")
        print(f"Wilcoxon signed-rank test on F1-Macro: p-value = {wilc_p_val_f1:.4f}")
        
        t_stat_acc, p_val_acc = ttest_rel(wsv_fold_accs, roberta_fold_accs)
        print(f"Paired t-test on Accuracy: statistic = {t_stat_acc:.4f}, p-value = {p_val_acc:.4f}")
        
        stats_results = {
            "mcnemar": {
                "statistic": float(stat),
                "p_value": float(p_value_mcnemar),
                "contingency_table": [[int(a), int(b)], [int(c), int(d)]]
            },
            "fold_wise": {
                "roberta_f1": [float(x) for x in roberta_fold_f1s],
                "wsv_f1": [float(x) for x in wsv_fold_f1s],
                "roberta_acc": [float(x) for x in roberta_fold_accs],
                "wsv_acc": [float(x) for x in wsv_fold_accs],
                "paired_t_test_f1": {"statistic": float(t_stat_f1), "p_value": float(p_val_f1)},
                "wilcoxon_f1": {"p_value": float(wilc_p_val_f1)},
                "paired_t_test_acc": {"statistic": float(t_stat_acc), "p_value": float(p_val_acc)}
            }
        }
        
        # Save results
        json_out = BASE_DIR / "model_civic_bodies" / "dataset_v2" / "ensemble_stacking" / "hypothesis_test_results.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(stats_results, f, indent=4)
        print(f"\nSaved statistical results to: {json_out}")
        
        # Plot contingency table and fold-wise metrics
        plt.figure(figsize=(10, 4), facecolor="white")
        
        # Subplot 1: Contingency Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["RoBERTa Correct", "RoBERTa Incorrect"],
                    yticklabels=["WSV Correct", "WSV Incorrect"])
        plt.title("Prediction Contingency Matrix", fontsize=10, fontweight="bold")
        
        # Subplot 2: Fold-wise Macro F1
        plt.subplot(1, 2, 2)
        df_plot = pd.DataFrame({
            "Fold": [f"Fold {i}" for i in range(5)] * 2,
            "Model": ["RoBERTa"] * 5 + ["Weighted Soft Voting"] * 5,
            "Macro F1": roberta_fold_f1s + wsv_fold_f1s
        })
        sns.barplot(data=df_plot, x="Fold", y="Macro F1", hue="Model", palette="Set2")
        plt.ylim(0.65, 0.80)
        plt.title("Fold-wise Macro F1 Comparison", fontsize=10, fontweight="bold")
        plt.tight_layout()
        
        plot_path = OUTPUT_DIR / "3.19_hypothesis_test_comparison.png"
        plt.savefig(plot_path, dpi=300, facecolor="white")
        plt.close()
        print(f"Saved visualization to: {plot_path}")

if __name__ == "__main__":
    main()
