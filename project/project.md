# Grievance IQ — Project System Design & Architecture

This document describes the technical blueprint, algorithmic progression, ensembling design, and system architecture of **Grievance IQ**. It details how the machine learning pipelines were trained, optimized, validated, and scaled to production.

---

## 1. System Engineering Blueprint

Grievance IQ serves as a production-grade automated routing and prioritization system for civic complaints. It integrates a **Weighted Soft-Voting Transformer Ensemble** for routing and a **Sequence-to-Sequence + Classification Pair Pipeline** for severity prioritization.

```text
               [ Raw Grievance Complaint ]
                            │
                            ▼
                [ NLTK Preprocessing ]
                            │
                            ▼
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
 [ Civic Agency Routing ]         [ Severity Prioritization ]
 ├── DistilBERT (OOF Probs)       ├── T5-Base CoT Reasoner
 ├── RoBERTa (OOF Probs)          │   └── (Generates explanation)
 └── DeBERTa v3 (OOF Probs)       ▼
          │                [ RoBERTa Text-Pair Head ]
          ▼                   └── (Complaint + T5 Reason)
[ Blended Soft Voting ]                       │
 (0.45 / 0.25 / 0.30)                         ▼
          │                       [ Priority Severity Level ]
          ▼                         (Critical/High/Med/Low)
[ Routed Department ]                         │
          │                                   │
          └─────────────────┬─────────────────┘
                            ▼
                 [ FastAPI Telemetry logs ]
                            │
                            ▼
               [ Glassmorphic Analytics ]
```

---

## 2. Phase 1 — Data cleaning & KDE Distribution Outlier Filtering

The raw dataset fetched from Supabase PostgreSQL (16,107 records) was cleaned and prepared through a rigorous pipeline:
* **Department Normalization**: Consolidated agency acronyms (e.g. mapping varied string variants to BBMP, BWSSB, BESCOM, etc.) and pruned ultra-sparse categories to reduce label entropy.
* **Length KDE Analysis**: Plotted the Kernel Density Estimate (KDE) of word counts across severity levels to find outliers. We established a lower bound of 1 word and an upper bound of 108 words (covering the 95th percentile) to truncate long-tail complaints and eliminate noise.
* **Text Preprocessing**: Implemented custom lowercasing, regex-based URL and special character removal, stopword filtering, and lemmatization using NLTK's `WordNetLemmatizer`.

---

## 3. Phase 2 — Civic Agency Classification (Transformer ensembling)

### 3.1 Base Model Folds (5-Fold CV)
Three deep learning architectures were fine-tuned using 5-Fold Cross Validation on Vertex AI to generate Out-of-Fold (OOF) prediction probabilities:
1. **DistilBERT** (`distilbert-base-uncased`): Light, high-throughput model. (Best fold F1-Macro: **0.7286**)
2. **RoBERTa** (`roberta-base`): High-capacity classifier with dynamic masking. (Best fold F1-Macro: **0.7279**)
3. **DeBERTa v3** (`microsoft/deberta-v3-base`): Advanced disentangled attention layers. (Best fold F1-Macro: **0.7221**)

### 3.2 Complementarity Analysis
We evaluated prediction overlaps to determine sample-level disagreement:
* **consensus (All 3 correct)**: 88.1% of samples.
* **irrecoverable (All 3 wrong)**: 4.4% of samples.
* **rescuable (Disagreement - at least one model correct)**: **7.5% (1,208 samples)**.
* **Oracle Ceiling**: **95.65% accuracy** / **0.8393 F1-Macro**, showing significant headroom for ensembling.

### 3.3 Soft Blending Optimization (Weighted Soft Voting)
To capture model confidence signals, we extracted softmax probability arrays and optimized blending weights. We evaluated three approaches:
1. **Hard-Label Stacking**: Meta-model (Logistic Regression) trained on one-hot features. Underperformed (**0.7217 F1-Macro**) due to loss of probability scale.
2. **Majority Voting**: Simple consensus baseline. (**0.7534 F1-Macro**, **+0.025** over best single model).
3. **Weighted Soft Voting (WSV)**: **WINNER**. Blended probability vectors using weights optimized via grid search:
   - DistilBERT Weight: **0.45**
   - RoBERTa Weight: **0.25**
   - DeBERTa v3 Weight: **0.30**
   * **Final Score**: **93.06% Accuracy** / **0.7576 F1-Macro** (**+0.029** macro F1 gain).

### 3.4 McNemar Statistical Hypothesis Testing
To confirm the validity of the ensembling gain, we ran McNemar's paired chi-squared test comparing the WSV ensemble against the single best model (RoBERTa):
* **Chi-Squared Statistic**: **62.77**
* **p-value**: **$2.34 \times 10^{-15}$**
The p-value is far below $\alpha = 0.05$, confirming that the ensembled model's performance improvements are highly statistically significant.

---

## 4. Phase 3 — Severity Prioritization (T5 + RoBERTa Pair Classifier)

Classifying complaint severity is difficult due to subjectiveness. To improve accuracy, we developed a two-stage **Chain-of-Thought (CoT)** pipeline:

1. **Stage 1 — Explanation Generation (T5-Base)**:
   Fine-tuned `t5-base` on a dataset of complaints and human-labeled severity reasons. On inference, the T5 model takes the raw complaint and generates a contextual explanation of why it belongs to a certain severity level.
2. **Stage 2 — Text-Pair Classification (RoBERTa)**:
   We feed the raw complaint concatenated with the generated T5 reason as a sentence pair into a custom `RoBERTa` classifier head (Fine-tuned on sentence pairs).
   * **Result**: The text-pair classifier reached **76.11% Accuracy / 0.7209 Macro F1**, outperforming classical regressors, standard DistilBERT regressors, and raw text classifiers by over **5.2%**.

---

## 5. Serving Layer & API Engineering

The backend was built using **FastAPI** to support production ensembling and telemetry:
* **Lifespan Initialization**: Uses Starlette's `lifespan` context manager to load all five neural networks into GPU/CPU memory once on server start, isolating setup overhead from request times.
* **Dual Loading (Cloud Fallback)**: Checks if weight folders exist locally (for local development/offline testing). If missing, it uses `huggingface_hub` client functions to pull and cache folds dynamically from the `sandipanarnab/grievance-iq-models` repository.
* **Testing Isolation**: Implements a dedicated `MOCK_MODELS` environment switch. When active, it bypasses model loading and inserts mock classes (returning mock tensors matching the correct shapes), allowing unit tests to run fully offline on GitHub runner environments.
* **Asynchronous Telemetry Logging**: Incoming predictions are appended to JSON Lines logs (`logs/complaints.jsonl`) via FastAPI `BackgroundTasks` to avoid delaying responses to the user.

---

## 6. Staging & Git-Isolated Deployment Pipeline

To deploy the codebase to Hugging Face Spaces without exceeding Git LFS limit rules, we created a decoupled build script:
1. **GitHub Actions Runner**: Executes unit tests in mock mode using `python -m pytest`.
2. **Isolated Staging Area**: The pipeline creates a temporary directory `deploy_temp/`, copying only production files (`src/`, `requirements.txt`, `Dockerfile`, `.dockerignore`, `README.md`).
3. **Fresh Git Init**: Initializes a fresh, history-less repository inside `deploy_temp/`. Since `models/` checkpoints, documentation graphs (`charts_and_graphs/`), and guides (`*.pdf`) are excluded, the staging folder is code-only (<100 KB).
4. **Push**: Force-pushes the clean staging folder to Hugging Face Spaces using the repository's actions secret `HF_TOKEN`.
