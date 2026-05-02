from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    complaint: str = Field(..., min_length=1, max_length=1000)

    @field_validator("complaint")
    @classmethod
    def validate_complaint(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Complaint must not be empty.")
        return cleaned


class PredictionResponse(BaseModel):
    predicted_department: str
    severity: Literal["critical", "high", "medium", "low"]
