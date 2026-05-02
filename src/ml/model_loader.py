from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPARTMENT_MODEL_DIR = PROJECT_ROOT / "metrics" / "linearsvc"
SEVERITY_MODEL_DIR = PROJECT_ROOT / "metrics_model_severity" / "distilbert" / "production_model_1"
SEVERITY_MODEL_FALLBACK_DIR = PROJECT_ROOT / "metrics_model_severity" / "distilbert" / "production_model"


@dataclass
class ModelBundle:
    department_model: object
    department_model_path: Path
    severity_model: AutoModelForSequenceClassification
    severity_tokenizer: AutoTokenizer
    severity_label_encoder: object
    severity_device: str
    severity_model_path: Path


def _latest_joblib_file(model_dir: Path) -> Path:
    files = list(model_dir.glob("*.joblib"))
    if not files:
        raise FileNotFoundError(f"No .joblib file found in {model_dir}")

    def version_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_v(\d+)\.joblib$", path.name)
        version = int(match.group(1)) if match else -1
        return version, path.name

    return sorted(files, key=version_key)[-1]


def _resolve_severity_model_dir() -> Path:
    if SEVERITY_MODEL_DIR.exists():
        return SEVERITY_MODEL_DIR
    if SEVERITY_MODEL_FALLBACK_DIR.exists():
        return SEVERITY_MODEL_FALLBACK_DIR
    raise FileNotFoundError(
        "Severity production model folder not found. "
        f"Checked {SEVERITY_MODEL_DIR} and {SEVERITY_MODEL_FALLBACK_DIR}."
    )


def load_model_bundle() -> ModelBundle:
    department_model_path = _latest_joblib_file(DEPARTMENT_MODEL_DIR)
    department_bundle = joblib.load(department_model_path)
    department_model = department_bundle["model"] if isinstance(department_bundle, dict) else department_bundle

    severity_model_path = _resolve_severity_model_dir()
    severity_tokenizer = AutoTokenizer.from_pretrained(severity_model_path)
    severity_model = AutoModelForSequenceClassification.from_pretrained(severity_model_path)
    severity_label_encoder = joblib.load(severity_model_path / "label_encoder.joblib")

    severity_device = "cuda" if torch.cuda.is_available() else "cpu"
    severity_model.to(severity_device)
    severity_model.eval()

    return ModelBundle(
        department_model=department_model,
        department_model_path=department_model_path,
        severity_model=severity_model,
        severity_tokenizer=severity_tokenizer,
        severity_label_encoder=severity_label_encoder,
        severity_device=severity_device,
        severity_model_path=severity_model_path,
    )
