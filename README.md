# AI Grievance System

Project 1: AI-driven citizen grievance severity labeling and NLP pipeline.

## Status: Data Labeling Completed
The initial data labeling phase using the Gemini API (`gemini-3.1-flash lite`) has been **successfully completed**. The `severity` labels and associated datasets have been generated and the manual review process is concluded.

## Data Collection (Current Phase)
We are now connecting directly to a Supabase PostgreSQL database to fetch the consolidated, labeled data for the next phase of our NLP and EDA pipelines.

### Setup Database Connection

1. Provide your connection details. The project expects a `.env` file in the `src/` directory with your Supabase credentials:

   ```env
   user=postgres
   password=your_supabase_password
   host=db.your_project_ref.supabase.co
   port=5432
   dbname=postgres
   ```

2. Please ensure you have the required dependencies for database connectivity. If missing, install them:
   ```powershell
   pip install sqlalchemy psycopg2-binary python-dotenv
   ```

3. Initial Phase: Explore, Clean, and Train (Jupyter Notebook)

   During the initial stage of the project, **all work should be done directly in a Jupyter Notebook** (`notebook/ai_grievance_system.ipynb`). In the notebook, your workflow will be:
   - Connect to the Supabase SQL database.
   - Fetch the data using SQL commands.
   - Analyze, preprocess/clean the data, and train models.

   Make sure to import these core libraries in your notebook to start:

   ```python
   import pandas as pd
   import numpy as np
   from sqlalchemy import create_engine
   from dotenv import load_dotenv
   import os
   ```

4. Final Production Stage (`src/main.py`):

   *Note: The standalone `src/main.py` script comes into play much later when the exploratory project is done and ready for deployment.* Once your notebook analysis is finalized, you can test and transition to the production data connection script from the root:
   ```powershell
   .\.venv\Scripts\python.exe src\main.py
   ```

   If successful, you should see:
   ```text
   Connection successful!
   ```

## Exploratory Data Analysis (EDA) & Preprocessing

The exploratory data analysis and data preprocessing phase is currently implemented in `notebook/ai_grievance_system.ipynb`. Key steps include:
- **Data Cleaning:** Discarding null values in critical features such as `description` and `severity`, and sorting records chronologically to track data drift.
- **Trend Analysis:** Comparing complaints over the years across severity categories. Key insights reveal that 'Medium' and 'High' severity incidents account for the majority of the volume.
- **Entity Standardization:** Standardizing civic agency names to prevent fragmentation (e.g., merging acronyms and full names for `BBMP`, `BTP`, `BWSSB`, `KSPCB`, and `BESCOM`).
- **Category Consolidation:** Grouping low-frequency entries to address class imbalance, such as merging `BDA` into `BBMP` and consolidating `BMTC` and `KSRTC` under a general "Transport" category.

## Model Training Pipeline

Following preprocessing, the notebook features a robust model training loop to automate severity scoring and department routing. The pipeline evaluates the following models:
1. **Logistic Regression**
2. **Linear Support Vector Classifier (LinearSVC)**
3. **Random Forest Classifier**

Models are trained iteratively using the processed textual data from the Supabase PostgreSQL database.

### Model Performance Metrics

Based on the cross-validation evaluation metrics tracked during the training phase, **LinearSVC** currently yields the best overall balance (highest Accuracy and Macro F1) for severity classification.

| Model | Accuracy | Macro F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **LinearSVC** | 90.41% | 0.774 | 0.753 | 0.803 |
| **Random Forest Classifier** | 89.60% | 0.734 | 0.824 | 0.683 |
| **Logistic Regression** | 89.12% | 0.762 | 0.726 | 0.809 |

## Next Steps in Pipeline

With EDA and initial training phases underway, the forthcoming goals include:
- **FastAPI Layer:** Serving the finalized models via API endpoints for seamless system integration.

---

## Folder Structure

```text
ai_grievance_system/
  data/                   Dataset files
    augmented_combined.csv
  docs/                   Documentation
  metrics/                Model evaluation metrics and results
    bert_base/
    linearsvc/
    logistic_regression/
    random_forest/
  notebook/               Jupyter notebooks for EDA and modeling
    ai_grievance_system.ipynb
  src/                    Source code and scripts
    .env                  Supabase credentials
    .env.example          Example credentials file
    main.py               Database connection testing and data fetching
  theory/                 Project references and theoretical docs
```

## Historical Context: AI Labeling Pipeline
*For reference, earlier stages of this project relied on the Gemini API to generate soft-labels.*
The labeling pipeline was designed as a checkpointed batch process that read raw complaints from `data/original/Complaints.csv`, labeled them using the `gemini-3.1-flash lite` model, and checkpointed 500-row chunks in `data/processed`. Ambiguous edges were placed in `data/review_bucket` for human alignment. This data is now upstreamed/synchronized to Supabase.
