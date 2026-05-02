## Project 1 Notes for Collaborators

These notes are meant to keep the team aligned on branch hygiene, local setup, notebook usage, and the FastAPI deployment flow.

## 1. Getting Started

Clone the repository and move into the project directory:

```bash
git clone https://github.com/sandipan-ds/ai_grievance_system.git
cd ai_grievance_system
```

Create your own working branch from `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/<your-name>
```

Use a clear branch name that reflects your work.

## 1.1 Important Run Commands

```powershell
pip install -r requirements.txt
uvicorn src.main:app --reload
streamlit run streamlit.py
```

Use the `uvicorn` command for the FastAPI backend and `/docs` testing. Use the Streamlit command only if you want the UI.

Main backend URLs after startup:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

## 2. Day-to-Day Workflow

Make your edits locally, then commit and push your branch:

```bash
git add .
git commit -m "Describe your changes"
git push -u origin feature/<your-name>
```

After pushing, open a pull request on GitHub:

- Base branch: `main`
- Compare branch: `feature/<your-name>`
- Review changes before merging

## 3. Keeping Your Branch Updated

If `main` changes while you are working, merge those updates into your feature branch:

```bash
git checkout main
git pull origin main
git checkout feature/<your-name>
git merge main
```

## 4. Collaboration Rules

- Never commit directly to `main`.
- Always work from a personal feature branch.
- Always open a pull request before merging.
- Keep commits small and descriptive when possible.

## 5. Supabase Database Setup

The project connects to a central Supabase PostgreSQL dataset through a local `.env` file.

1. Locate `src/.env.example`.
2. Rename it to `src/.env`.
3. Fill in your Supabase credentials.
4. Make sure any local path values point to your actual cloned project directory.

Example:

```env
user=postgres
password=your_supabase_password
host=db.your_project_ref.supabase.co
port=5432
dbname=postgres
```

## 6. How To Use the Project

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Then follow the main working flow:

1. Use `notebook/ai_grievance_system.ipynb` for exploration, preprocessing, augmentation, cross-validation, and model training.
2. Use the saved `joblib` model artifacts under `metrics/` for inference.
3. Run the FastAPI backend with:

```powershell
uvicorn src.main:app --reload
```

4. Open Swagger at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/openapi.json`

5. Test the backend using `POST /predict` with:

```json
{
  "complaint": "Garbage is overflowing near my house and the road is full of potholes."
}
```

You can also test from PowerShell:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"complaint":"Garbage is overflowing near my house and the road is full of potholes."}'
```

The backend returns:

```json
{
  "predicted_department": "BBMP",
  "severity": "medium"
}
```

Validation notes:

- `complaint` must not be empty
- `complaint` must be at most `1000` characters

6. If needed, run the Streamlit app separately:

```powershell
streamlit run streamlit.py
```

## 7. FastAPI Backend Notes

The API entry point is `src/main.py`.

The backend loads:

- civic-agency prediction from `metrics/linearsvc/`
- severity prediction from `metrics_model_severity/distilbert/production_model_1/`

Prediction logs are appended to:

- `logs/complaints.jsonl`

Each logged record includes:

- `id`
- `timestamp`
- `complaint`
- `predicted_department`
- `severity`
- `model_version`

The database utilities used by the backend now live in:

- `src/db/postgres.py`

## 8. Notes for the Notebook Pipeline

Most of the exploratory work, preprocessing, augmentation, cross-validation, and model training lives in:

- `notebook/ai_grievance_system.ipynb`

Use the notebook for experimentation first, then move stable logic into `src/` or `streamlit.py` when the workflow is ready for production-style use.
