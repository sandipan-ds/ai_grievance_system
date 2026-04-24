from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Avoid shadowing the Streamlit package with this file name.
SCRIPT_DIR = Path(__file__).resolve().parent
_clean_path = []
for entry in sys.path:
    try:
        resolved = Path(entry or ".").resolve()
    except Exception:
        _clean_path.append(entry)
        continue
    if resolved != SCRIPT_DIR:
        _clean_path.append(entry)
sys.path = _clean_path

import streamlit as st


PROJECT_ROOT = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
METRICS_DIR = PROJECT_ROOT / "metrics"
NLTK_DIR = DATA_DIR / "nltk"

MODEL_REGISTRY = {
    "LinearSVC": METRICS_DIR / "linearsvc",
    "Logistic Regression": METRICS_DIR / "logistic_regression",
    "Random Forest": METRICS_DIR / "random_forest",
}

MODEL_OPTIONS = [
    "LinearSVC",
    "Logistic Regression",
    "Random Forest",
    "All",
]

LOOKUP_MAX_ROW = 17145


def _ensure_nltk_paths() -> None:
    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DIR))


def _build_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return {
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
            "for", "from", "has", "have", "he", "her", "his", "i", "in", "is",
            "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
            "the", "their", "them", "there", "these", "they", "this", "to",
            "was", "were", "we", "with", "you", "your",
        }


_ensure_nltk_paths()
STOPWORDS = _build_stopwords()
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in text.split():
        try:
            lemma = LEMMATIZER.lemmatize(token)
        except LookupError:
            lemma = token
        if lemma not in STOPWORDS:
            tokens.append(lemma)

    return " ".join(tokens)


def _latest_joblib_file(model_dir: Path) -> Path:
    files = list(model_dir.glob("*.joblib"))
    if not files:
        raise FileNotFoundError(f"No .joblib file found in {model_dir}")

    def version_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_v(\d+)\.joblib$", path.name)
        version = int(match.group(1)) if match else -1
        return version, path.name

    return sorted(files, key=version_key)[-1]


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_name: str):
    model_dir = MODEL_REGISTRY[model_name]
    model_path = _latest_joblib_file(model_dir)
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], model_path
    return bundle, model_path


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "augmented_combined.csv",
        usecols=["description", "civic_agency_title"],
    )
    return df


def predict_with_model(model_name: str, raw_text: str) -> str:
    model, _ = load_model_bundle(model_name)
    cleaned = preprocess_text(raw_text)
    if not cleaned.strip():
        raise ValueError("Complaint text is empty after preprocessing.")
    return str(model.predict([cleaned])[0])


def predict_all_models(raw_text: str) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_REGISTRY:
        prediction = predict_with_model(model_name, raw_text)
        rows.append({"Model": model_name, "Predicted Department": prediction})
    return pd.DataFrame(rows)


def render_prediction_area(raw_text: str, selected_model: str, include_actual: bool = False, actual_label: str | None = None) -> None:
    predictions = predict_all_models(raw_text) if selected_model == "All" else pd.DataFrame(
        [{"Model": selected_model, "Predicted Department": predict_with_model(selected_model, raw_text)}]
    )

    st.markdown("### Output")

    if include_actual and actual_label is not None:
        st.markdown(f"**Actual department:** {actual_label}")

    if selected_model == "All":
        st.dataframe(predictions, use_container_width=True)
        unique_predictions = predictions["Predicted Department"].nunique()
        if unique_predictions == 1:
            st.success(f"All models agree on: {predictions['Predicted Department'].iloc[0]}")
        else:
            st.info("The models do not fully agree. Use the table above to compare predictions.")
    else:
        prediction = predictions["Predicted Department"].iloc[0]
        st.success(f"Predicted department: {prediction}")


def render_model_details() -> None:
    with st.sidebar:
        st.subheader("Loaded Models")
        for model_name in MODEL_REGISTRY:
            _, model_path = load_model_bundle(model_name)
            st.write(f"{model_name}: {model_path.name}")
        st.caption("Text preprocessing: lowercase, URL removal, stopword removal, lemmatization.")
        st.caption("Dataset lookup range: 0 to 17145.")


def main() -> None:
    st.set_page_config(
        page_title="AI Grievance System",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
        .subtle {
                color: #666666;
                margin-top: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">AI Grievance System</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">Test civic-department predictions for new complaints or inspect existing dataset rows.</p>',
        unsafe_allow_html=True,
    )

    render_model_details()

    dataset = load_dataset()
    max_row = min(LOOKUP_MAX_ROW, len(dataset) - 1)

    tab_new, tab_dataset = st.tabs(["New Complaint Testing", "Dataset Complaint Lookup"])

    with tab_new:
        st.subheader("New Complaint Testing")
        st.write("Enter a fresh complaint and compare the civic-department prediction across models.")

        with st.form("new_complaint_form"):
            complaint_text = st.text_area(
                "Complaint text",
                placeholder="Type the grievance or complaint here...",
                height=220,
            )
            selected_model = st.selectbox("Model", MODEL_OPTIONS, index=3)
            submitted = st.form_submit_button("Predict")

        if submitted:
            if not complaint_text.strip():
                st.warning("Please enter a complaint before predicting.")
            else:
                render_prediction_area(complaint_text, selected_model)

    with tab_dataset:
        st.subheader("Dataset Complaint Lookup")
        st.write("Use a dataset row number to inspect the original complaint, actual department, and model predictions.")
        st.caption(f"Valid row numbers: 0 to {max_row}")

        with st.form("dataset_lookup_form"):
            row_number = st.number_input(
                "Row number",
                min_value=0,
                max_value=max_row,
                value=0,
                step=1,
            )
            selected_model_ds = st.selectbox("Model", MODEL_OPTIONS, index=3, key="dataset_model")
            lookup = st.form_submit_button("Lookup")

        if lookup:
            if row_number < 0 or row_number > max_row:
                st.error(f"Row number must be between 0 and {max_row}.")
            else:
                row = dataset.iloc[int(row_number)]
                complaint_text = str(row["description"])
                actual_label = str(row["civic_agency_title"])

                st.markdown("### Complaint Text")
                st.text_area("description", value=complaint_text, height=240, disabled=True)

                render_prediction_area(
                    complaint_text,
                    selected_model_ds,
                    include_actual=True,
                    actual_label=actual_label,
                )


if __name__ == "__main__":
    main()
