# AI Grievance System

Project 1: AI-driven citizen grievance severity labeling and NLP pipeline.

## Status: Data Labeling Completed
The initial data labeling phase using local LLMs (Ollama) has been **successfully completed**. The `severity` labels and associated datasets have been generated and the manual review process is concluded.

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

3. Test the Database Connection:

   Run the data connection script from the project root:
   ```powershell
   .\.venv\Scripts\python.exe src\main.py
   ```

   If successful, you should see:
   ```text
   Connection successful!
   ```

## Next Steps in Pipeline

With database connectivity established and labeling completed, the next phases cover:
- **Text Preprocessing & EDA:** Using the newly integrated PostgreSQL data pool to perform extensive NLP analyses.
- **Model Training:** Training supervised models for automated severity scoring and department routing.
- **FastAPI Layer:** Serving the models.

---

## Folder Structure

```text
ai_grievance_system/
  data/
    original/             (Archived raw CSVs prior to labeling)
    processed/            (Completed labeled chunk files)
    review_bucket/        (Completed manual review CSVs)
  docs/                   Documentation
  notebook/               Jupyter notebooks for EDA
  src/
    .env                  Supabase credentials
    main.py               Database connection testing and data fetching
```

## Historical Context: Local AI Labeling Pipeline
*For reference, earlier stages of this project relied on a local Ollama setup to generate soft-labels.*
The labeling pipeline was designed as a checkpointed batch process that read raw complaints from `data/original/Complaints.csv`, labeled them with a local Ollama model (e.g., `llama3.3`), and checkpointed 500-row chunks in `data/processed`. Ambiguous edges were placed in `data/review_bucket` for human alignment. This data is now upstreamed/synchronized to Supabase.
