# Project 1 Notes

Project 1 is an AI-driven citizen grievance and sentiment analysis system. The larger roadmap includes text preprocessing and EDA, department or civic-agency routing, urgency or severity scoring, model evaluation, and a FastAPI serving layer.

## Current State: Database Connection & Data Collection

The initial challenge of the project was that `data/original/Complaints.csv` lacked a `severity` label. The AI Severity labeling pipeline was built to fix this and has now been **completed**. 

Currently, the project focuses on **Data Collection**. Having finished local labeling, we have connected the workflow to a **Supabase PostgreSQL** database. The database now acts as the system's central data warehouse for the labeled records, which will be fetched directly for downstream EDA and model building.

## Supabase PostgreSQL Connection

We connect to Supabase utilizing SQLAlchemy and `psycopg2`. 

- **Connection Script:** `src/main.py`
- **Environment configuration:** `src/.env` (contains `user`, `password`, `host`, `port`, `dbname`). Note: Ensure the SSL mode is set to require as per standard Supabase connection strings.

Fetching data via `main.py` ensures that all subsequent NLP and Data Analysis routines work off a single unified, up-to-date data source, avoiding local CSV fragmentation.

---

## Historical: Labeling Design (Completed)

*The following details how the baseline severity labels were originally obtained using LLMs. This phase is complete.*

- Raw input: `data/original/Complaints.csv`
- Output chunks: `data/processed/labeled_dataset_1.csv`, `data/processed/labeled_dataset_2.csv`, etc.
- Human review queue: CSV files at `data/review_bucket/review_bucket_1.csv`, etc.
- Chunk size: 500 rows
- Model provider: local Ollama API (`llama3.3`)

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
