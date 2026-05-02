# AI Grievance System

Project 1: a multi-output, multiclass NLP system for citizen grievance handling.

## Quick Start

```powershell
pip install -r requirements.txt
streamlit run streamlit.py
```

Optional database connection test:

```powershell
.\.venv\Scripts\python.exe src\main.py
```

The project takes a complaint `description` as input and aims to:

- route the complaint to the correct civic authority
- assign the correct severity class

The currently implemented pipeline focuses on two core tasks:
1. **Authority-routing model**: Predicting the correct civic department using traditional ML models.
2. **Severity classification model**: Predicting the severity (Critical, High, Medium, Low) using a custom PyTorch LSTM neural network with Focal Loss and a penalty matrix to handle class imbalance.

## Project Status

- Initial severity data labeling was completed using the Gemini API.
- Complaint data is now consolidated in Supabase PostgreSQL.
- The notebook pipeline covers EDA, preprocessing, augmentation, cross-validation, and model training for both authority routing and severity classification.
- Final authority model artifacts are saved as `joblib` files, while the PyTorch LSTM models are saved as `.pt` weights. All models and metrics are tracked with DVC.

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

### 3. Launch the Streamlit app

Run the interactive app from the project root:

```powershell
streamlit run streamlit.py
```

The app lets you:

- type a new complaint and get a predicted civic department
- look up a complaint by dataset row number
- compare predictions across the saved LinearSVC, Logistic Regression, and Random Forest models

### 4. Optional production connection test

If you want to test the database connection script:
If the connection is correct, it should print `Connection successful!`.

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

The notebook trains models for both prediction tasks:

**Authority Routing:**
Trains multiple classical text classifiers using a TF-IDF pipeline:
1. Logistic Regression
2. LinearSVC
3. Random Forest Classifier

**Severity Classification:**
Trains a custom LSTM-based neural network using PyTorch. The training loop includes:
- Differentiable focal loss function with a cost-sensitive penalty matrix to mitigate class imbalance
- Early stopping based on validation F1-Macro score
- Learning rate scheduler (`ReduceLROnPlateau`)

The best hyperparameters are evaluated using 5-fold cross-validation.

### 9. Artifact Storage

The project stores:

- trained classical model files as `joblib`
- trained PyTorch LSTM weights as `.pt` files
- tuning results, evaluation summaries, and confusion matrices in `metrics/` (authority) and `metrics_model_severity/` (severity)
- versioned data and model artifacts with DVC

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
- model training
- metrics generation

## Streamlit App

The interactive UI is in:

- `streamlit.py`

It provides two workflows:

- **New Complaint Testing** for typing a fresh complaint and viewing predictions
- **Dataset Complaint Lookup** for inspecting an existing complaint row and comparing the actual department with model predictions

The app uses the saved `.joblib` model artifacts in `metrics/` and the processed complaint data from `data/`. It loads the latest saved model bundle for each algorithm automatically.

The dataset used for inference is expected to be available as `data/augmented_combined.csv`, with `description` as the complaint text input and `civic_agency_title` as the target department label.

## Folder Structure

```text
ai_grievance_system/
  data/                   Dataset files and processed artifacts
  docs/                   Documentation and project notes
  metrics/                Model metrics, confusion matrices, and saved models (Authority)
  metrics_model_severity/ Model metrics and saved PyTorch model weights (Severity)
  notebook/               EDA and training notebook
  src/                    Source code, database connection, and env files
  theory/                 Supporting references and theory notes
  streamlit.py            Streamlit application entrypoint
```

## Historical Context

Earlier stages of the project used the Gemini API to generate severity labels from raw complaint data. That labeling pipeline ran in checkpointed batches, wrote processed outputs incrementally, and sent ambiguous cases for review. The consolidated labeled data now feeds the Supabase-backed workflow used in the notebook.
