from __future__ import annotations

import re
from pathlib import Path

import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.inference.model_loader import ModelBundle

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


# Class mappings
CIVIC_AGENCY_LABELS = [
    "BBMP",
    "BCP",
    "BESCOM",
    "BTP",
    "BWSSB",
    "KSFES",
    "KSPCB",
    "Transport",
]

SEVERITY_CLASSES = [
    "Non-Grievance",
    "Low",
    "Medium",
    "High",
    "Critical",
]


def preprocess_text(text: str) -> str:
    """Cleans and tokenizes text for model inference."""
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
    """
    Predicts the target department using Weighted Soft Voting (WSV) across
    DistilBERT, RoBERTa, and DeBERTa v3.
    Weights: DistilBERT=0.45, RoBERTa=0.25, DeBERTa v3=0.30
    """
    cleaned = preprocess_text(complaint)
    if not cleaned:
        raise ValueError("Complaint text is empty after preprocessing.")

    device = bundle.device

    # DistilBERT
    inputs_distil = bundle.distilbert_tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=256
    )
    inputs_distil = {k: v.to(device) for k, v in inputs_distil.items()}

    # RoBERTa Civic
    inputs_rob = bundle.roberta_civic_tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=256
    )
    inputs_rob = {k: v.to(device) for k, v in inputs_rob.items()}

    # DeBERTa v3
    inputs_deb = bundle.deberta_tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=256
    )
    inputs_deb = {k: v.to(device) for k, v in inputs_deb.items()}

    with torch.inference_mode():
        outputs_distil = bundle.distilbert_model(**inputs_distil)
        outputs_rob = bundle.roberta_civic_model(**inputs_rob)
        outputs_deb = bundle.deberta_model(**inputs_deb)

        probs_distil = torch.softmax(outputs_distil.logits, dim=-1)
        probs_rob = torch.softmax(outputs_rob.logits, dim=-1)
        probs_deb = torch.softmax(outputs_deb.logits, dim=-1)

        # Apply optimized ensembling soft voting weights
        blended_probs = 0.45 * probs_distil + 0.25 * probs_rob + 0.30 * probs_deb
        pred_idx = torch.argmax(blended_probs, dim=-1).item()

    return CIVIC_AGENCY_LABELS[pred_idx]


def predict_severity(bundle: ModelBundle, complaint: str) -> str:
    """
    Predicts complaint severity in a two-stage sequential pipeline:
    1. T5 model generates the severity reason from the complaint description.
    2. RoBERTa classifier evaluates the description + reason pair to produce the class.
    """
    cleaned = preprocess_text(complaint)
    if not cleaned:
        raise ValueError("Complaint text is empty after preprocessing.")

    device = bundle.device

    # ── Stage 1: T5 Reason Generation ─────────────────────────────────────────
    # The T5 model expects input prefix: "predict severity reason: "
    t5_input_text = "predict severity reason: " + cleaned
    t5_inputs = bundle.t5_tokenizer(
        t5_input_text, return_tensors="pt", truncation=True, max_length=512
    )
    t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}

    with torch.inference_mode():
        generated_ids = bundle.t5_model.generate(**t5_inputs, max_length=128)
        
    generated_reason = bundle.t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_reason = generated_reason.strip()

    # ── Stage 2: RoBERTa Classification ───────────────────────────────────────
    # Tokenize as a text pair: description text + generated severity reason
    classification_inputs = bundle.severity_tokenizer(
        cleaned,
        text_pair=generated_reason,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    classification_inputs = {k: v.to(device) for k, v in classification_inputs.items()}

    with torch.inference_mode():
        logits = bundle.severity_classifier(
            input_ids=classification_inputs["input_ids"],
            attention_mask=classification_inputs["attention_mask"],
        )
        pred_idx = torch.argmax(logits, dim=-1).item()

    # Return severity label (normalize to lowercase to keep interface consistent)
    return SEVERITY_CLASSES[pred_idx].lower()


def predict_complaint(bundle: ModelBundle, complaint: str) -> tuple[str, str]:
    """Classifies department and severity for a given complaint."""
    department = predict_department(bundle, complaint)
    severity = predict_severity(bundle, complaint)
    return department, severity
