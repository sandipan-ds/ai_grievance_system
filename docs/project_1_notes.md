# Project 1 Notes

Project 1 is an AI-driven citizen grievance and sentiment analysis system. The larger roadmap includes text preprocessing and EDA, department or civic-agency routing, urgency or severity scoring, model evaluation, and a FastAPI serving layer.

For the current dataset, `data/original/Complaints.csv` already contains department-style routing fields such as `category_title`, `sub_category_title`, and `civic_agency_title`. The immediate missing label is `severity`, so the first practical step is to create a reproducible labeling pipeline before training any supervised severity model.

## Current Labeling Design

- Raw input: `data/original/Complaints.csv`
- Sample input: `data/original/sample_dataset.csv`
- Output chunks: `data/processed/labeled_dataset_1.csv`, `data/processed/labeled_dataset_2.csv`, etc.
- Human review rows: CSV files at `data/review_bucket/review_bucket_1.csv`, `data/review_bucket/review_bucket_2.csv`, etc.
- Chunk size: 500 rows by default
- Label column: `severity`
- Audit columns: `confidence_score`, `reason_high_confidence`, `reason_ambiguity`, `needs_review`, `labeling_error`, `raw_model_response`
- Human review columns in review bucket only: `human_reviewed`, `human_severity`, `review_notes`
- Labels: `Critical`, `High`, `Medium`, `Low`
- Model provider: local Ollama API
- Default model: `llama3.3`

The labeling runner checks existing numbered chunk files in `data/processed` and resumes from the next missing complete chunk. For example, if `labeled_dataset_1.csv` and `labeled_dataset_2.csv` exist, the next run starts from row 1001 of `Complaints.csv` and writes `labeled_dataset_3.csv`.

Rows are copied to `data/review_bucket` when the model fails to return a valid severity or when `confidence_score` is below the review threshold. The default threshold is `0.80`. These review files stay in CSV format because they are meant for human correction in spreadsheets, pandas, or notebooks. This keeps the main processed dataset complete while creating a smaller queue for human inspection.

## Recommended Structure

```text
ai_grievance_system/
  data/
    original/
      Complaints.csv
      sample_dataset.csv
    processed/
      labeled_dataset_1.csv
      labeled_dataset_2.csv
    review_bucket/
      review_bucket_1.csv
  docs/
    project_1_notes.md
  notebook/
    ai_grievance_system.ipynb
  src/
    labeling/
      clients/
        ollama_client.py
      config.py
      io/
        checkpoints.py
      main.py
      ollama.py
      pipeline.py
      prompts/
        severity.py
```

Yes, the original `main.py` or all-in-one labeling script should be divided into smaller files. It makes the prompt easier to tune, the Ollama call easier to debug, and the checkpoint logic safer to reuse when the full 16,071-row dataset is labeled.
