from __future__ import annotations

import re
from pathlib import Path

import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.ml.model_loader import ModelBundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NLTK_DIR = PROJECT_ROOT / "data" / "nltk"


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

    tokens: list[str] = []
    for token in text.split():
        try:
            lemma = LEMMATIZER.lemmatize(token)
        except LookupError:
            lemma = token
        if lemma not in STOPWORDS:
            tokens.append(lemma)

    return " ".join(tokens)


def predict_department(bundle: ModelBundle, complaint: str) -> str:
    cleaned = preprocess_text(complaint)
    if not cleaned:
        raise ValueError("Complaint text is empty after preprocessing.")
    return str(bundle.department_model.predict([cleaned])[0])


def predict_severity(bundle: ModelBundle, complaint: str) -> str:
    cleaned = preprocess_text(complaint)
    if not cleaned:
        raise ValueError("Complaint text is empty after preprocessing.")

    inputs = bundle.severity_tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(bundle.severity_device)

    with torch.inference_mode():
        outputs = bundle.severity_model(**inputs)

    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    predicted_label = bundle.severity_label_encoder.inverse_transform([predicted_class_id])[0]
    return str(predicted_label).lower()


def predict_complaint(bundle: ModelBundle, complaint: str) -> tuple[str, str]:
    department = predict_department(bundle, complaint)
    severity = predict_severity(bundle, complaint)
    return department, severity
