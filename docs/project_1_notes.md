# Project 1 Notes

Project 1 is an AI-driven citizen grievance and sentiment analysis system. The larger roadmap includes text preprocessing and EDA, department or civic-agency routing, urgency or severity scoring, model evaluation, and a FastAPI serving layer.

## Current State: Database Connection & Data Collection

The initial challenge of the project was that `data/original/Complaints.csv` lacked a `severity` label. The AI Severity labeling pipeline was built to fix this and has now been **completed**. 

Currently, the project focuses on **Data Collection**. Having finished local labeling, we have connected the workflow to a **Supabase PostgreSQL** database. The database now acts as the system's central data warehouse for the labeled records, which will be fetched directly for downstream EDA and model building.

## Supabase PostgreSQL Connection

We connect to Supabase utilizing SQLAlchemy and `psycopg2`. 

- **Environment configuration:** `src/.env` (contains `user`, `password`, `host`, `port`, `dbname`). Note: Ensure the SSL mode is set to require as per standard Supabase connection strings.
- **Production Connection Script:** `src/main.py`

**Important Workflow Note: Jupyter Notebook First**
In the initial and primary stages of the project, **all tasks** should be performed exclusively inside a Jupyter Notebook (`notebook/ai_grievance_system.ipynb`).

The correct workflow is to:
1. Connect to the Supabase SQL database inside the notebook.
2. Fetch the data using standard SQL commands.
3. Perform Exploratory Data Analysis (EDA), preprocess and clean the data.
4. Train and validate NLP models.

The standalone `src/main.py` connection script is **reserved for much later** in the project, serving as the entry point when the team is ready to package and deploy the final pipeline.

To start fetching data and exploring in your notebook, ensure you run these imports:

```python
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
```

Fetching data directly from the Supabase connection during the EDA phase ensures that all analysis routines work off a single unified, up-to-date data source, avoiding local CSV fragmentation.

---

## Historical: Labeling Design (Completed)

*The following details how the baseline severity labels were originally obtained using LLMs. This phase is complete.*

- Raw input: `data/original/Complaints.csv`
- Output chunks: `data/processed/labeled_dataset_1.csv`, `data/processed/labeled_dataset_2.csv`, etc.
- Human review queue: CSV files at `data/review_bucket/review_bucket_1.csv`, etc.
- Chunk size: 500 rows
- Model provider: Gemini API (`gemini-3.1-flash lite`)

The labeling runner checked existing numbered chunk files in `data/processed` and resumed from the next missing complete chunk. Rows were copied to `data/review_bucket` when the model failed to return a valid severity or when `confidence_score` fell below the review threshold (default `0.80`). These review files stayed in CSV format for human correction.

## Recommended Folder Structure

```text
ai_grievance_system/
  data/
    original/
    processed/        (Historical chunks)
    review_bucket/    (Historical manual reviews)
  docs/
    project_1_notes.md
  notebook/
    ai_grievance_system.ipynb
  src/
    .env              (Database environment variables)
    main.py           (Database connection and data retrieval)
```
