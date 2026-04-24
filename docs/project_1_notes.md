## Project 1 Notes for Collaborators

These notes are meant to keep the team aligned on branch hygiene, local setup, notebook usage, and the Streamlit app.

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
streamlit run streamlit.py
.\.venv\Scripts\python.exe src\main.py
```

Use the Streamlit command for the app, and the Python command for a database connection check.

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
3. Run the Streamlit app with:

```powershell
streamlit run streamlit.py
```

The Streamlit app supports:

- typing a new complaint and getting a predicted civic department
- checking an existing dataset row by row number
- comparing predictions across LinearSVC, Logistic Regression, and Random Forest

## 7. Notes for the Notebook Pipeline

Most of the exploratory work, preprocessing, augmentation, cross-validation, and model training lives in:

- `notebook/ai_grievance_system.ipynb`

Use the notebook for experimentation first, then move stable logic into `src/` or `streamlit.py` when the workflow is ready for production-style use.
