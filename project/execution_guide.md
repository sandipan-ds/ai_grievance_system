# Grievance IQ — End-to-End Execution Guide

Grievance IQ is a production-grade MLOps system that automatically routes civic grievances to the correct department and assigns a priority severity level using an ensembled deep learning pipeline. 

This guide details the monorepo architecture and provides the step-by-step instructions required to run the entire pipeline from raw data extraction to live containerized cloud deployment.

---

## 1. Project Architecture & Pipeline Flow

The system processes incoming complaints sequentially through the following modules:

```text
Complaint Text
      │
      ▼
[ 1. Text Preprocessing ] ────► Lowercasing, URL/special char removal, NLTK Lemmatization
      │
      ▼
[ 2. Agency Classifier ] ─────► Weighted Soft-Voting Ensemble (DistilBERT + RoBERTa + DeBERTa v3)
      │
      ▼
[ 3. Reason Generator ] ──────► T5-Base Sequence-to-Sequence (Generates Severity Reason Chain-of-Thought)
      │
      ▼
[ 4. Severity Classifier ] ───► Custom RoBERTa Text-Pair Head (Classifies Complaint + T5 Reason)
      │
      ▼
[ 5. FastAPI Backend ] ───────► Asynchronous prediction, operational logging, and real-time analytics
      │
      ▼
[ 6. Glassmorphic UI ] ───────► Real-time Tailwind CSS SPA with Chart.js analytics logs
```

---

## 2. Environment Setup

### 2.1 Install Dependencies
Initialize a virtual environment and install the required Python packages:
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Upgrade pip and install packages
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Configure Local Environment Variables
Create a `.env` file at the root of the repository (`c:\Users\sandi\Desktop\ML Working Folder\ai_grievance_system\.env`) containing your credentials:
```env
# Supabase PostgreSQL Database Credentials
user=your_db_username
password=your_db_password
host=your_db_host.supabase.co
port=5432
dbname=postgres

# Hugging Face Settings
HF_TOKEN=hf_your_write_access_token
```

---

## 3. End-to-End Execution Steps

### Step 3.1: Data Preprocessing & EDA
Run the Jupyter notebook `notebooks/ai_grievance_system_fixed.ipynb` to execute:
1. **Section 1**: Connects to Supabase, pulls raw complaint records, cleans null values, deduplicates text, and standardizes civic department labels.
2. **Section 2**: Runs Exploratory Data Analysis, fits word-length KDE curves, crops outliers, and exports distribution graphs to `charts_and_graphs/`.

---

### Step 3.2: Model Training (5-Fold CV OOF)
To train the individual classifiers on Vertex AI (or locally), execute the training scripts. Each script fine-tunes a model on the 5 folds and saves the probability Out-of-Fold (OOF) vectors:

```bash
# 1. DistilBERT Civic Agency Classifier
python scripts/distilbert_train_civic_v2.py
# Submit Vertex AI Job:
python scripts/submit_vertex_job_civic_v2.py

# 2. RoBERTa Civic Agency Classifier
python scripts/roberta_train_civic_v2.py
# Submit Vertex AI Job:
python scripts/submit_vertex_job_civic_roberta_v2.py

# 3. DeBERTa v3 Civic Agency Classifier
python scripts/deberta_v3_train_civic_v2.py
# Submit Vertex AI Job:
python scripts/submit_vertex_job_civic_deberta_v3.py
```

---

### Step 3.3: Blended Soft Stacking Ensemble
Once training completes, extract the probability vectors and run the optimization ensembling process:

```bash
# 1. Extract and cache probability arrays locally from GCS/local runs
python scripts/download_results_extract_probs.py

# 2. Execute Weighted Soft Voting optimization
python scripts/ensemble_soft_stacking.py
```
* **Output**: Calculates the best ensembling weights (**DistilBERT=0.45, RoBERTa=0.25, DeBERTa v3=0.30**), raising F1-Macro to **0.7576** (accuracy **93.06%**). It exports heatmap graphs to `charts_and_graphs/civic_agency_results/`.

---

### Step 3.4: Statistical Significance Verification
Validate that the ensembled model's performance improvements are statistically significant using McNemar's Chi-Squared test:
```bash
python tests/hypothesis_testing.py
```
* **Output**: Computes contingency tables and a p-value of **$2.34 \times 10^{-15}$**, demonstrating high statistical significance.

---

### Step 3.5: Severity & Reason Generation Models
Train the severity reasoning model and text-pair classifier:
```bash
# 1. Train T5-base Severity Reason Generator
python scripts/T5_base_train_reason.py

# 2. Train RoBERTa Text-Pair Classifier (Tuned on Complaint + generated reason)
python scripts/roberta_classifier_severity_v2.py
```

---

### Step 3.6: Upload Best Checkpoints to Model Hub Registry
Since Hugging Face Spaces limits deployment repositories to 1 GB, we store the heavy model weights (2.5 GB+) in a dedicated Model Hub.
Run the upload utility to create the repository and push folds:
```bash
python upload_models.py
```
* **Target Repository**: `sandipanarnab/grievance-iq-models`

---

## 4. Serving the Application (Local Runtime)

### 4.1 Running FastAPI Dashboard Locally
Run the server locally. On startup, the dual loader will check for local weights inside `models/`; if absent, it pulls them dynamically from the Hugging Face Model Hub registry:

```bash
# Serves the app at http://localhost:7860
python -m uvicorn src.inference.main:app --host 0.0.0.0 --port 7860
```

*To run fast, lightweight testing without downloading model weights, set the mock environment variable:*
```bash
# On Windows:
$env:MOCK_MODELS="1"; python -m uvicorn src.inference.main:app --host 0.0.0.0 --port 7860
# On Linux/macOS:
MOCK_MODELS=1 python -m uvicorn src.inference.main:app --host 0.0.0.0 --port 7860
```

### 4.2 serving via Docker Locally
```bash
# Build the production Docker image (excludes notebooks and tests)
docker build -t grievance-iq:latest .

# Run the container exposing port 7860
docker run -p 7860:7860 grievance-iq:latest
```

---

## 5. Automated CI/CD Pipeline (GitHub Actions)

We run two isolated workflows on every push/pull-request to the `main` or `master` branches of the repository:

### 5.1 CI Unit Testing (`pytest.yml`)
Runs the full suite of unit tests in mock mode (bypassing model downloads and database integrations to execute in seconds).
* **Command**: `python -m pytest tests/`

### 5.2 Deployment Pipeline (`deploy.yml`)
Once the unit tests pass:
1. Creates a clean staging directory on the GitHub Actions runner.
2. Copies only the deployment assets (`src/`, `requirements.txt`, `Dockerfile`, `.dockerignore`, `README.md`).
3. Initializes a fresh git repository inside the staging area, stripping out documentation graphs, PDFs, and binary history.
4. Authenticates using your repository's actions secret `HF_TOKEN` and force-pushes the clean code to Hugging Face Spaces.
