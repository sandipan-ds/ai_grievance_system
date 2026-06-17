from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    RobertaModel,
    RobertaTokenizer,
)

# Root directory configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths to the model directories (best folds based on cross-validation metrics)
DISTILBERT_DIR = PROJECT_ROOT / "models" / "civic_bodies" / "dataset_v2" / "DistilBERT" / "fold_3" / "model"
ROBERTA_CIVIC_DIR = PROJECT_ROOT / "models" / "civic_bodies" / "dataset_v2" / "RoBERTa" / "fold_3" / "model"
DEBERTA_CIVIC_DIR = PROJECT_ROOT / "models" / "civic_bodies" / "dataset_v2" / "DeBERTa_v3" / "fold_3" / "model"

T5_SEVERITY_DIR = PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "t5_base_reason" / "fold_0" / "model"
ROBERTA_SEVERITY_DIR = PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "trial_5_roberta_classifier" / "fold_0" / "model"


class RoBERTaClassifier(nn.Module):
    """Custom PyTorch module matching the definition in the training notebook (Trial 5)."""
    def __init__(self, model_name="roberta-base", dropout=0.1, num_classes=5, is_mock=False):
        super().__init__()
        self.is_mock = is_mock
        if not is_mock:
            self.roberta = RobertaModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        else:
            # Setup a minimal linear layer for matching weights structure
            self.classifier = nn.Linear(10, num_classes)

    def forward(self, input_ids, attention_mask):
        if self.is_mock:
            return torch.zeros((input_ids.shape[0], 5), dtype=torch.float32)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


@dataclass
class ModelBundle:
    # Civic Agency models (WSV)
    distilbert_model: AutoModelForSequenceClassification | object
    distilbert_tokenizer: AutoTokenizer | object
    roberta_civic_model: AutoModelForSequenceClassification | object
    roberta_civic_tokenizer: AutoTokenizer | object
    deberta_model: AutoModelForSequenceClassification | object
    deberta_tokenizer: AutoTokenizer | object
    
    # Severity models
    t5_model: T5ForConditionalGeneration | object
    t5_tokenizer: T5Tokenizer | object
    severity_classifier: RoBERTaClassifier
    severity_tokenizer: RobertaTokenizer | object
    
    # Device configuration
    device: str
    is_mock: bool


class MockTokenizer:
    """Mock tokenizer to avoid HF network downloads during CI/unit testing."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, text, text_pair=None, **kwargs):
        seq_len = 16
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
        return {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }

    def encode(self, *args, **kwargs):
        return [1, 2, 3]

    def decode(self, *args, **kwargs):
        return "mocked generated severity reason"

    def batch_decode(self, sequences, *args, **kwargs):
        return ["mocked generated severity reason" for _ in sequences]


class MockModel(nn.Module):
    """Mock model to avoid loading heavy weights during CI/unit testing."""
    def __init__(self, num_labels=8):
        super().__init__()
        self.num_labels = num_labels
        self.config = type("Config", (), {"hidden_size": 768})()

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        class Outputs:
            def __init__(self, num_labels, batch_size):
                self.logits = torch.zeros((batch_size, num_labels), dtype=torch.float32)
                # Assign highest logit to label 0
                self.logits[:, 0] = 5.0
        return Outputs(self.num_labels, batch_size)

    def generate(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        return torch.zeros((batch_size, 5), dtype=torch.long)


def load_model_bundle() -> ModelBundle:
    """Loads all models. Automatically loads mock models if in a CI/testing environment."""
    # Check if we should use mock models (useful for fast pytest runs on GitHub Actions without GCS credentials)
    use_mock = "pytest" in sys.modules or os.getenv("MOCK_MODELS") == "1"

    if use_mock:
        print("[INFO] Model loader: Pytest/Mock environment detected. Loading lightweight mock models.")
        return ModelBundle(
            distilbert_model=MockModel(num_labels=8),
            distilbert_tokenizer=MockTokenizer(),
            roberta_civic_model=MockModel(num_labels=8),
            roberta_civic_tokenizer=MockTokenizer(),
            deberta_model=MockModel(num_labels=8),
            deberta_tokenizer=MockTokenizer(),
            t5_model=MockModel(),
            t5_tokenizer=MockTokenizer(),
            severity_classifier=RoBERTaClassifier(is_mock=True),
            severity_tokenizer=MockTokenizer(),
            device="cpu",
            is_mock=True,
        )

    # Determine CPU or GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Model loader: Initializing production models on device '{device}'...")

    hf_repo = "sandipanarnab/grievance-iq-models"

    # Load Civic Agency models (Fold 3)
    if DISTILBERT_DIR.exists():
        distilbert_tokenizer = AutoTokenizer.from_pretrained(
            str(DISTILBERT_DIR / "tokenizer") if (DISTILBERT_DIR / "tokenizer").exists() else "distilbert-base-uncased",
            use_fast=False,
        )
        distilbert_model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR)).to(device)
    else:
        print(f"[INFO] Model loader: Local DistilBERT not found. Fetching from HF Hub '{hf_repo}'...")
        distilbert_tokenizer = AutoTokenizer.from_pretrained(
            hf_repo,
            subfolder="model_civic_bodies/dataset_v2/DistilBERT/fold_3/tokenizer",
            use_fast=False,
        )
        distilbert_model = AutoModelForSequenceClassification.from_pretrained(
            hf_repo,
            subfolder="model_civic_bodies/dataset_v2/DistilBERT/fold_3/model",
        ).to(device)
    distilbert_model.eval()

    if ROBERTA_CIVIC_DIR.exists():
        roberta_civic_tokenizer = AutoTokenizer.from_pretrained(
            str(ROBERTA_CIVIC_DIR / "tokenizer") if (ROBERTA_CIVIC_DIR / "tokenizer").exists() else "roberta-base",
            use_fast=False,
        )
        roberta_civic_model = AutoModelForSequenceClassification.from_pretrained(str(ROBERTA_CIVIC_DIR)).to(device)
    else:
        print(f"[INFO] Model loader: Local RoBERTa civic not found. Fetching from HF Hub '{hf_repo}'...")
        roberta_civic_tokenizer = AutoTokenizer.from_pretrained(
            hf_repo,
            subfolder="model_civic_bodies/dataset_v2/RoBERTa/fold_3/tokenizer",
            use_fast=False,
        )
        roberta_civic_model = AutoModelForSequenceClassification.from_pretrained(
            hf_repo,
            subfolder="model_civic_bodies/dataset_v2/RoBERTa/fold_3/model",
        ).to(device)
    roberta_civic_model.eval()

    if DEBERTA_CIVIC_DIR.exists():
        deberta_tokenizer = AutoTokenizer.from_pretrained(
            str(DEBERTA_CIVIC_DIR / "tokenizer") if (DEBERTA_CIVIC_DIR / "tokenizer").exists() else "microsoft/deberta-v3-base",
            use_fast=False,
        )
        deberta_model = AutoModelForSequenceClassification.from_pretrained(str(DEBERTA_CIVIC_DIR)).to(device)
    else:
        print(f"[INFO] Model loader: Local DeBERTa civic not found. Fetching from HF Hub '{hf_repo}'...")
        deberta_tokenizer = AutoTokenizer.from_pretrained(
            hf_repo,
            subfolder="model_civic_bodies/dataset_v2/DeBERTa_v3/fold_3/tokenizer",
            use_fast=False,
        )
        deberta_model = AutoModelForSequenceClassification.from_pretrained(
            hf_repo,
            subfolder="model_civic_bodies/dataset_v2/DeBERTa_v3/fold_3/model",
        ).to(device)
    deberta_model.eval()

    # Load T5 Severity Reason Generator (Fold 0)
    t5_local_path = None
    if (PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "t5_base_reason" / "model").exists():
        t5_local_path = PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "t5_base_reason" / "model"
    elif (PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "t5_base_reason" / "fold_0" / "model").exists():
        t5_local_path = PROJECT_ROOT / "models" / "severity" / "dataset_v2" / "t5_base_reason" / "fold_0" / "model"

    if t5_local_path is not None:
        print(f"[INFO] Model loader: Loading local T5 severity model from '{t5_local_path}'...")
        t5_tokenizer_path = t5_local_path.parent / "tokenizer"
        t5_tokenizer = T5Tokenizer.from_pretrained(
            str(t5_tokenizer_path) if t5_tokenizer_path.exists() else "t5-base",
            use_fast=False,
        )
        t5_model = T5ForConditionalGeneration.from_pretrained(str(t5_local_path)).to(device)
    else:
        print(f"[INFO] Model loader: Local T5 severity not found. Fetching from HF Hub '{hf_repo}'...")
        t5_tokenizer = T5Tokenizer.from_pretrained(
            hf_repo,
            subfolder="model_severity/dataset_v2/t5_base_reason/fold_0/tokenizer",
            use_fast=False,
        )
        t5_model = T5ForConditionalGeneration.from_pretrained(
            hf_repo,
            subfolder="model_severity/dataset_v2/t5_base_reason/fold_0/model",
        ).to(device)
    t5_model.eval()

    # Load RoBERTa Severity Classifier (Fold 0)
    if ROBERTA_SEVERITY_DIR.exists():
        severity_tokenizer = RobertaTokenizer.from_pretrained(
            str(ROBERTA_SEVERITY_DIR / "tokenizer") if (ROBERTA_SEVERITY_DIR / "tokenizer").exists() else "roberta-base",
            use_fast=False,
        )
        severity_classifier = RoBERTaClassifier("roberta-base", num_classes=5).to(device)
        state_dict = torch.load(ROBERTA_SEVERITY_DIR / "pytorch_model.pt", map_location=device)
    else:
        print(f"[INFO] Model loader: Local RoBERTa severity classifier not found. Fetching from HF Hub '{hf_repo}'...")
        severity_tokenizer = RobertaTokenizer.from_pretrained(
            hf_repo,
            subfolder="model_severity/dataset_v2/trial_5_roberta_classifier/fold_0/model/tokenizer",
            use_fast=False,
        )
        severity_classifier = RoBERTaClassifier("roberta-base", num_classes=5).to(device)
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id=hf_repo,
            filename="model_severity/dataset_v2/trial_5_roberta_classifier/fold_0/model/pytorch_model.pt",
        )
        state_dict = torch.load(weights_path, map_location=device)
    
    severity_classifier.load_state_dict(state_dict)
    severity_classifier.eval()

    print("[INFO] Model loader: All production models loaded successfully.")

    return ModelBundle(
        distilbert_model=distilbert_model,
        distilbert_tokenizer=distilbert_tokenizer,
        roberta_civic_model=roberta_civic_model,
        roberta_civic_tokenizer=roberta_civic_tokenizer,
        deberta_model=deberta_model,
        deberta_tokenizer=deberta_tokenizer,
        t5_model=t5_model,
        t5_tokenizer=t5_tokenizer,
        severity_classifier=severity_classifier,
        severity_tokenizer=severity_tokenizer,
        device=device,
        is_mock=False,
    )
