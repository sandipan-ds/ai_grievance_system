# AI Grievance System

Project 1: a multi-output, multiclass NLP system for citizen grievance handling.

## Quick Start

```powershell
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Then open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- FastAPI OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

Quick API test from PowerShell:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"complaint":"Garbage is overflowing near my house and the road is damaged."}'
```

The project takes a complaint `description` as input and aims to:

- route the complaint to the correct civic authority
- assign the correct severity class

The project now includes both:

- a civic-agency routing pipeline
- a severity prediction pipeline

The FastAPI backend serves both predictions together from the same complaint text.

## Project Status

- Initial data labeling has been completed using the Gemini API.
- Complaint data is now consolidated in Supabase PostgreSQL.
- The notebook pipeline covers EDA, preprocessing, augmentation, cross-validation, and model training.
- Final model artifacts are saved as `joblib` files and tracked with DVC.

## How To Use

### 1. Set up the environment

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Create `src/.env` with your Supabase credentials:

```env
user=postgres
password=your_supabase_password
host=db.your_project_ref.supabase.co
port=5432
dbname=postgres
```

### 2. Run the data / model notebook

Open and run:

- `notebook/ai_grievance_system.ipynb`

This notebook is where the data loading, preprocessing, augmentation, cross-validation, tuning, and model training logic lives.

For severity modeling specifically, use **Section 5** of the notebook.

That section covers:

- severity-specific preprocessing
- augmentation and fold preparation
- DistilBERT training and evaluation
- recall-focused DistilBERT experiments for `Critical`
- LSTM training and evaluation
- classical baselines: Logistic Regression, LinearSVC, and Random Forest

### 3. Launch the FastAPI backend

Run the API from the project root:

```powershell
uvicorn src.main:app --reload
```

Swagger will be available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

### 4. Test the prediction endpoint

Use the Swagger UI or send a POST request to:

- `POST /predict`

Request body:

```json
{
  "complaint": "Garbage is overflowing near my house and the road is full of potholes."
}
```

Response shape:

```json
{
  "predicted_department": "BBMP",
  "severity": "medium"
}
```

PowerShell test example:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"complaint":"Street lights are not working and the road is unsafe at night."}'
```

Validation rules for the request:

- `complaint` must not be empty
- maximum length is `1000` characters

Every successful prediction is appended to:

- `logs/complaints.jsonl`

Each log entry contains:

- `id`
- `timestamp`
- `complaint`
- `predicted_department`
- `severity`
- `model_version`

### 5. Optional Streamlit UI

If you still want the interactive Streamlit interface:

```powershell
streamlit run streamlit.py
```

## Workflow Overview

### 1. Data Collection

The notebook connects to Supabase PostgreSQL and loads the complaint dataset for analysis and modeling.

### 2. Data Cleaning

The raw data is cleaned before training:

- remove null rows from key fields such as `description` and target columns
- remove duplicates
- standardize date fields for time-based analysis

### 3. Exploratory Data Analysis

The notebook explores the dataset to understand complaint patterns and class imbalance:

- complaint volume over time
- distribution across severity classes
- distribution across civic agencies, categories, and sub-categories
- complaint concentration in major authorities and rare classes

### 4. Text Preprocessing

Complaint text is normalized using a standard NLP cleaning pipeline:

- lowercase conversion
- URL removal
- special character and non-letter removal
- stopword removal
- lemmatization

### 5. Agency Standardization and Consolidation

To reduce label fragmentation, civic agency names are standardized and low-frequency groups are merged where appropriate.

Examples:

- `Bruhat Bengaluru Mahanagara Palike` -> `BBMP`
- `Bangalore Traffic Police` -> `BTP`
- `BDA` -> `BBMP`
- `BMTC` and `KSRTC` -> `Transport`

### 6. Data Augmentation

To reduce class imbalance, complaint text is augmented based on sample frequency:

- low-frequency classes receive more augmentation
- higher-frequency classes receive less augmentation

The augmentation strategy includes synonym replacement, spelling perturbation, and word-level shuffling.

### 7. Cross-Validation and Hyperparameter Search

Hyperparameters are selected using 5-fold cross-validation.

The process is:

- split the preprocessed dataset into 5 folds
- use 4 folds for training
- augment only the training folds
- validate on the untouched original fold

This approach reduces repeated preprocessing, keeps validation data clean, and gives a more realistic estimate of model performance.

### 8. Model Training

The notebook trains multiple classical text classifiers using a TF-IDF pipeline:

1. Logistic Regression
2. LinearSVC
3. Random Forest Classifier

The best hyperparameters are saved into the `metrics/` folder, and the final model is trained from those tuned settings.

### 9. Artifact Storage

The project stores:

- trained model files as `joblib`
- tuning results and evaluation summaries in `metrics/`
- versioned data and model artifacts with DVC

## Severity Training And Evaluation

The severity workflow is implemented in **Section 5** of:

- `notebook/ai_grievance_system.ipynb`

### Severity preprocessing

The severity models use the complaint `description` field as input and `severity` as the target.

The notebook applies:

- lowercasing
- URL removal
- non-letter / special-character cleanup
- whitespace normalization
- stopword removal
- lemmatization

The same section also builds severity-specific augmented folds and saves them as:

- `data/processed/cv_fold_data_augmented_severity.joblib`

These folds follow the same CV policy used elsewhere in the project:

- split the cleaned data into 5 folds
- use 4 folds for training
- augment only the training portion
- validate on the untouched original fold

### Severity models trained in the notebook

The notebook trains and evaluates:

1. DistilBERT baseline
2. DistilBERT recall-focused variant for `Critical`
3. LSTM
4. Logistic Regression
5. LinearSVC
6. Random Forest

### How severity evaluation is saved

Depending on the model family, the notebook writes out:

- per-fold metrics JSON files
- classification reports
- `summary.json` files with mean and standard deviation across folds
- aggregated or out-of-fold confusion matrix images

Main output folders:

- `metrics_model_severity/distilbert/metrics_model_1/`
- `metrics_model_severity/distilbert/model_2/`
- `metrics_model_severity/lstm/native/`
- `metrics_model_severity/logistic_regression/`
- `metrics_model_severity/linearsvc/`
- `metrics_model_severity/random_forest/`

### Severity results summary

From the saved evaluation artifacts:

| Model | Accuracy | Macro F1 | Notes |
| :--- | :--- | :--- | :--- |
| **DistilBERT (baseline)** | 0.79 | 0.73 | Best overall severity model in the saved metrics |
| **DistilBERT (recall-focused)** | 0.76 | 0.72 | Cost-sensitive experiment to reduce `Critical` under-calling |
| **Logistic Regression** | 0.75 | 0.68 | Strongest classical baseline among the saved severity runs |
| **LinearSVC** | 0.74 | 0.67 | Close classical baseline |
| **Random Forest** | 0.75 | 0.64 | Higher accuracy than macro F1 because minority classes are harder |
| **LSTM** | 0.64 | 0.60 | Lowest overall among the saved severity experiments |

### Current production severity model

The backend currently loads the exported severity model from:

- `metrics_model_severity/distilbert/production_model_1/`

That production folder contains the tokenizer, label encoder, config, and model weights used by the FastAPI `/predict` endpoint.

## Best Performing Model

Based on the tracked cross-validation results, `LinearSVC` provides the best overall balance of accuracy and macro F1 for the current authority-routing task.

| Model | Accuracy | Macro F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **LinearSVC** | 90.41% | 0.774 | 0.753 | 0.803 |
| **Random Forest Classifier** | 89.60% | 0.734 | 0.824 | 0.683 |
| **Logistic Regression** | 89.12% | 0.762 | 0.726 | 0.809 |

## Notebook

The main exploratory and modeling work lives in:

- `notebook/ai_grievance_system.ipynb`

That notebook contains:

- database access
- EDA
- preprocessing
- augmentation
- cross-validation
- civic-agency model training
- severity model training
- metrics generation

## Streamlit App

The interactive UI is in:

- `streamlit.py`

It provides two workflows:

- **New Complaint Testing** for typing a fresh complaint and viewing predictions
- **Dataset Complaint Lookup** for inspecting an existing complaint row and comparing the actual department with model predictions

The app uses the saved `.joblib` model artifacts in `metrics/` and the processed complaint data from `data/`. It loads the latest saved model bundle for each algorithm automatically.

## FastAPI Backend

The API backend lives in:

- `src/main.py`

The backend:

- loads the civic-agency and severity models once at startup
- validates input with Pydantic
- exposes `POST /predict`
- logs predictions asynchronously to `logs/complaints.jsonl`
- serves Swagger at `/docs`
- keeps database connection setup reusable through `src/db/postgres.py`

Backend structure:

- `src/main.py`
- `src/db/postgres.py`
- `src/ml/model_loader.py`
- `src/ml/predictor.py`
- `src/schemas/schemas.py`
- `src/logger/logger.py`

## Folder Structure

```text
ai_grievance_system/
  .dvc/                           DVC metadata
  .github/                        GitHub workflows and repo settings
  checkpoints/                    Training checkpoints for deep models
  data/
    nltk/                         Local NLTK resources
    processed/                    Processed datasets and fold artifacts
      cv_fold_data_augmented_civic_agencies.joblib
      cv_fold_data_augmented_severity.joblib
      cv_folds_augmented.joblib
      final_dataset.joblib
    augmented_combined.csv        Dataset used by the Streamlit lookup flow
  docs/
    project_1_notes.md            Project guide and collaborator notes
  logs/
    complaints.jsonl              FastAPI prediction logs
  metrics/
    linearsvc/                    Civic-agency LinearSVC model and metrics
    logistic_regression/          Civic-agency Logistic Regression metrics
    MultinomialNB/                Civic-agency MultinomialNB metrics
    random_forest/                Civic-agency Random Forest model and metrics
  metrics_model_severity/
    distilbert/
      metrics_model_1/            DistilBERT baseline severity evaluation
      model_2/                    Recall-focused DistilBERT evaluation
      production_model_1/         Exported severity model used by FastAPI
    linearsvc/                    Severity LinearSVC metrics
    logistic_regression/          Severity Logistic Regression metrics
    lstm/
      native/                     LSTM severity evaluation outputs
    multinomialnb/                Severity MultinomialNB metrics
    random_forest/                Severity Random Forest metrics
  notebook/
    ai_grievance_system.ipynb     Main EDA, preprocessing, training, and evaluation notebook
  src/
    db/
      postgres.py                 Reusable PostgreSQL / Supabase connection helpers
    logger/
      logger.py                   JSONL prediction logging
    ml/
      model_loader.py             Startup model loading
      predictor.py                Shared preprocessing and prediction logic
    schemas/
      schemas.py                  Pydantic request and response models
    .env.example                  Example backend environment config
    main.py                       FastAPI entrypoint
  theory/                         Supporting references and theory notes
  README.md                       Main project documentation
  requirements.txt                Python dependencies
  streamlit.py                    Streamlit application entrypoint
```

## Historical Context

Earlier stages of the project used the Gemini API to generate severity labels from raw complaint data. That labeling pipeline ran in checkpointed batches, wrote processed outputs incrementally, and sent ambiguous cases for review. The consolidated labeled data now feeds the Supabase-backed workflow used in the notebook.
