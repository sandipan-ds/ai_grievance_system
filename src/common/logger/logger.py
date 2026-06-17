from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE_PATH = PROJECT_ROOT / "logs" / "complaints.jsonl"


def append_prediction_log(
    complaint: str,
    predicted_department: str,
    severity: str,
    severity_reason: str = "",
    model_version: str = "v1.0",
) -> None:
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "complaint": complaint,
        "predicted_department": predicted_department,
        "severity": severity,
        "severity_reason": severity_reason,
        "model_version": model_version,
    }

    with LOG_FILE_PATH.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
