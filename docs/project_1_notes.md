## Project 1 Notes

This document is a practical guide for understanding, running, testing, and collaborating on the project in the right sequence.

## 1. Project Objective

This project is a multi-output, multiclass NLP system built on citizen complaint text.

The complaint `description` is used to predict:

- the correct civic authority or department
- the correct severity level

At the project level, there are two connected pipelines:

- civic-agency prediction
- severity prediction

The current application layer serves both predictions together through FastAPI.

## 2. Repository Structure

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
    project_1_notes.md            This project guide
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

## 3. Initial Setup

Clone the repository and enter the project folder:

```bash
git clone https://github.com/sandipan-ds/ai_grievance_system.git
cd ai_grievance_system
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create the environment file at `src/.env`.

Example:

```env
user=postgres
password=your_supabase_password
host=db.your_project_ref.supabase.co
port=5432
dbname=postgres
```

## 4. Important Run Commands

Client-facing use:

```powershell
streamlit run streamlit.py
```

Developer / Swagger use:

```powershell
uvicorn src.main:app --reload
```

Main backend URLs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

Quick PowerShell API test:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"complaint":"Garbage is overflowing near my house and the road is damaged."}'
```

Streamlit can auto-start the local FastAPI backend when it is not already running on `127.0.0.1:8000`, so the client-facing run command is usually enough for demos.

## 5. Data And Training Workflow

The main experimentation and model-building work is done in:

- `notebook/ai_grievance_system.ipynb`

The notebook flow is:

1. load complaint data from Supabase/PostgreSQL
2. clean the dataset
3. run EDA
4. preprocess complaint text
5. standardize agency labels where needed
6. augment minority classes
7. prepare 5-fold cross-validation data
8. tune and evaluate models
9. train final models and save artifacts

## 6. Preprocessing Logic

The notebook uses a standard complaint-text preprocessing pipeline:

- lowercase conversion
- URL removal
- special character and non-letter cleanup
- whitespace normalization
- stopword removal
- lemmatization

This preprocessing is reused in the inference pipeline as well.

## 7. Cross-Validation Logic

The project uses a fold strategy designed to keep validation realistic.

For each 5-fold split:

1. split the preprocessed dataset into 5 parts
2. select 4 folds for training
3. augment only the training folds
4. keep the validation fold untouched
5. train on augmented training data
6. validate on the original validation fold

This logic is used so that augmentation helps the model learn without contaminating validation.

## 8. Civic-Agency Prediction

The civic-agency task predicts the correct civic body from complaint text.

Models evaluated include:

- Logistic Regression
- LinearSVC
- Random Forest

The best-performing civic-agency model is the `LinearSVC` pipeline stored under:

- `metrics/linearsvc/`

This model is the department model used by the backend.

## 9. Severity Prediction

Severity modeling is handled in Section 5 of the notebook.

The severity target is predicted from the same complaint `description` field.

Severity models evaluated include:

- DistilBERT baseline
- recall-focused DistilBERT variant
- LSTM
- Logistic Regression
- LinearSVC
- Random Forest

Severity-specific fold data is stored at:

- `data/processed/cv_fold_data_augmented_severity.joblib`

Saved severity outputs live under:

- `metrics_model_severity/distilbert/metrics_model_1/`
- `metrics_model_severity/distilbert/model_2/`
- `metrics_model_severity/lstm/native/`
- `metrics_model_severity/logistic_regression/`
- `metrics_model_severity/linearsvc/`
- `metrics_model_severity/random_forest/`

The exported production severity model currently used by the backend is:

- `metrics_model_severity/distilbert/production_model_1/`

## 10. Evaluation Outputs

The training notebook saves evaluation artifacts such as:

- best-parameter JSON files
- per-fold metrics
- classification reports
- summary JSON files
- out-of-fold confusion matrices
- averaged confusion matrices for deep-learning models

These outputs are used to compare models before choosing the production version.

## 11. FastAPI Backend

The FastAPI entry point is:

- `src/main.py`

Supporting backend modules:

- `src/db/postgres.py`
- `src/ml/model_loader.py`
- `src/ml/predictor.py`
- `src/schemas/schemas.py`
- `src/logger/logger.py`

The backend:

- loads models once at startup
- validates requests with Pydantic
- exposes `POST /predict`
- returns both department and severity
- logs predictions asynchronously

Request shape:

```json
{
  "complaint": "string"
}
```

Response shape:

```json
{
  "predicted_department": "string",
  "severity": "critical|high|medium|low"
}
```

Validation rules:

- `complaint` must not be empty
- `complaint` must be at most `1000` characters

## 12. Logging

Successful predictions are appended to:

- `logs/complaints.jsonl`

Each record includes:

- `id`
- `timestamp`
- `complaint`
- `predicted_department`
- `severity`
- `model_version`

## 13. Streamlit App

The Streamlit interface is in:

- `streamlit.py`

It is now a client-facing front end that talks to the FastAPI backend instead of loading models locally.

It is useful for:

- testing new complaint text
- checking predictions against dataset rows
- viewing severity alongside civic-agency prediction

Streamlit uses:

- `POST /predict` for backend inference
- `data/augmented_combined.csv` for dataset lookup

The backend URL can be changed from the Streamlit sidebar if the API is hosted somewhere else.

## 14. DVC Usage

Large files are managed through DVC rather than regular Git tracking.

This includes items such as:

- processed datasets
- `.joblib` model files
- larger model artifacts

Git should mainly track lightweight files such as:

- source code
- markdown documentation
- JSON outputs
- images
- DVC pointer files

## 15. Recommended Working Sequence

If someone is new to the project, the clean order is:

1. set up the environment and `.env`
2. read the notebook to understand the pipeline
3. inspect `metrics/` and `metrics_model_severity/`
4. run the FastAPI backend
5. test `/predict` in Swagger or PowerShell
6. use Streamlit if interactive testing is needed
7. make changes in a feature branch

## 16. Collaboration Workflow

Create a personal working branch:

```bash
git checkout main
git pull origin main
git checkout -b feature/<your-name>
```

Commit and push your work:

```bash
git add .
git commit -m "Describe your changes"
git push -u origin feature/<your-name>
```

Then open a pull request into `main`.

Rules:

- do not commit directly to `main`
- keep commits clear and focused
- open a pull request before merging

If `main` changes while you are working:

```bash
git checkout main
git pull origin main
git checkout feature/<your-name>
git merge main
```
