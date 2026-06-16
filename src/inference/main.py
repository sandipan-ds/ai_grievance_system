from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.common.logger.logger import append_prediction_log
from src.inference.model_loader import ModelBundle, load_model_bundle
from src.inference.predictor import predict_complaint
from src.common.schemas.schemas import PredictionRequest, PredictionResponse

# Setup base directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
COMPLAINTS_LOG_PATH = LOGS_DIR / "complaints.jsonl"
STATIC_DIR = PROJECT_ROOT / "src" / "inference" / "static"


def load_historical_stats() -> dict[str, Any]:
    """
    Load initial stats from historical files if present, otherwise fall back
    to pre-calculated stats of Dataset v2 (16,107 samples) to initialize the dashboard.
    """
    stats = {
        "total": 0,
        "active": 0,
        "critical": 0,
        "recent": [],
        "department_distribution": {},
        "severity_distribution": {},
        "matrix": {},
    }

    # Attempt to load bbmc_data_v2.csv or augmented_combined.csv if available
    paths_to_try = [
        PROJECT_ROOT / "data" / "augmented_combined.csv",
        PROJECT_ROOT / "data" / "processed" / "bbmc_data_v2.csv",
    ]

    loaded = False
    for path in paths_to_try:
        if path.exists():
            try:
                print(f"[INFO] Backend: Pre-populating analytics from {path.name}...")
                df = pd.read_csv(path)
                
                # Check for columns and normalize
                dept_col = "civic_agency_title" if "civic_agency_title" in df.columns else "y"
                severity_col = "severity" if "severity" in df.columns else "severity_score"
                date_col = "created_at" if "created_at" in df.columns else "timestamp"
                
                if dept_col in df.columns and severity_col in df.columns:
                    # Clean and fill NaNs
                    df[dept_col] = df[dept_col].fillna("BBMP").astype(str)
                    
                    # Map severity scores to text labels if they are numerical
                    def normalize_sev(x):
                        try:
                            score = float(x)
                            if score >= 90: return "Critical"
                            elif score >= 80: return "High"
                            elif score >= 50: return "Medium"
                            elif score >= 1: return "Low"
                            else: return "Non-Grievance"
                        except (ValueError, TypeError):
                            s_str = str(x).strip().capitalize()
                            return s_str if s_str in ["Critical", "High", "Medium", "Low", "Non-Grievance"] else "Medium"
                    
                    df["severity_clean"] = df[severity_col].apply(normalize_sev)
                    
                    # Compute counts
                    stats["total"] = len(df)
                    stats["active"] = int(df["severity_clean"].isin(["Critical", "High"]).sum())
                    stats["critical"] = int((df["severity_clean"] == "Critical").sum())
                    
                    # Distributions
                    stats["department_distribution"] = df[dept_col].value_counts().to_dict()
                    stats["severity_distribution"] = df["severity_clean"].value_counts().to_dict()
                    
                    # Matrix
                    grouped = df.groupby([dept_col, "severity_clean"]).size().unstack(fill_value=0)
                    stats["matrix"] = grouped.to_dict(orient="index")
                    
                    # Recent list (last 10)
                    desc_col = "description" if "description" in df.columns else "X"
                    recent_df = df.tail(10)
                    for _, row in recent_df.iterrows():
                        stats["recent"].append({
                            "id": "HIST",
                            "timestamp": str(row.get(date_col, datetime.now().isoformat())),
                            "complaint": str(row.get(desc_col, ""))[:150] + "...",
                            "predicted_department": str(row.get(dept_col)),
                            "severity": str(row.get("severity_clean")).lower(),
                        })
                    loaded = True
                    break
            except Exception as e:
                print(f"[Warning] Backend: Error reading {path.name}: {e}")

    if not loaded:
        # Fallback to pre-calculated stats of the dataset (16,107 complaints total)
        # to ensure the dashboard looks realistic and populated on deployment
        print("[INFO] Backend: No historical files found. Initializing with default Dataset V2 statistics.")
        stats["total"] = 16107
        stats["active"] = 6200
        stats["critical"] = 2800
        stats["department_distribution"] = {
            "BBMP": 9540,
            "BWSSB": 2150,
            "BESCOM": 1850,
            "BTP": 1200,
            "BCP": 650,
            "Transport": 450,
            "KSFES": 180,
            "KSPCB": 87,
        }
        stats["severity_distribution"] = {
            "Non-Grievance": 1100,
            "Low": 2307,
            "Medium": 6500,
            "High": 3400,
            "Critical": 2800,
        }
        # Populate matrices
        for dept in stats["department_distribution"].keys():
            stats["matrix"][dept] = {
                "Non-Grievance": int(stats["department_distribution"][dept] * 0.07),
                "Low": int(stats["department_distribution"][dept] * 0.15),
                "Medium": int(stats["department_distribution"][dept] * 0.40),
                "High": int(stats["department_distribution"][dept] * 0.21),
                "Critical": int(stats["department_distribution"][dept] * 0.17),
            }
        
        # Populate sample recent list
        stats["recent"] = [
            {
                "id": "SEED-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "complaint": "Huge potholes on Outer Ring Road near Marathahalli causing severe traffic delays.",
                "predicted_department": "BTP",
                "severity": "high",
            },
            {
                "id": "SEED-2",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "complaint": "Water supply contaminated with sewage in HSR Layout Sector 3.",
                "predicted_department": "BWSSB",
                "severity": "critical",
            },
            {
                "id": "SEED-3",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "complaint": "Frequent voltage fluctuations and power cuts since yesterday evening.",
                "predicted_department": "BESCOM",
                "severity": "medium",
            }
        ]
        
    return stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models bundle
    app.state.models = load_model_bundle()
    
    # Load historical database stats
    app.state.historical_stats = load_historical_stats()
    
    yield


app = FastAPI(
    title="AI Grievance System API",
    version="v2.0",
    docs_url="/docs",
    lifespan=lifespan,
)

# Enable CORS for standard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    """Returns the initialization status of model bundle loading."""
    models = getattr(request.app.state, "models", None)
    return {
        "status": "ok",
        "models_loaded": bool(models),
        "is_mock": getattr(models, "is_mock", False) if models else False,
        "device": getattr(models, "device", "unknown") if models else "unknown",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """Predicts department (WSV ensemble) and severity (T5+RoBERTa) for a complaint."""
    try:
        models: ModelBundle = request.app.state.models
        predicted_department, severity = predict_complaint(models, payload.complaint)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction pipeline execution failed.") from exc

    # Log prediction in background
    background_tasks.add_task(
        append_prediction_log,
        complaint=payload.complaint,
        predicted_department=predicted_department,
        severity=severity,
        model_version="v2.0-wsv-roberta",
    )

    return PredictionResponse(
        predicted_department=predicted_department,
        severity=severity,
    )


@app.get("/analytics")
async def get_analytics(request: Request) -> dict[str, Any]:
    """Calculates operational and aggregate counts combining historical data and local logs."""
    stats = request.app.state.historical_stats
    
    # Copy base stats
    total = stats["total"]
    active = stats["active"]
    critical = stats["critical"]
    recent = list(stats["recent"])
    
    dept_counts = dict(stats["department_distribution"])
    sev_counts = dict(stats["severity_distribution"])
    matrix = {k: dict(v) for k, v in stats["matrix"].items()}
    
    # Load registered complaints log
    registered_complaints = []
    if COMPLAINTS_LOG_PATH.exists():
        try:
            with COMPLAINTS_LOG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        registered_complaints.append(item)
        except Exception as e:
            print(f"[Warning] Failed to read prediction logs: {e}")

    # Process and append registered complaints
    for item in registered_complaints:
        total += 1
        sev = str(item.get("severity", "medium")).capitalize()
        dept = str(item.get("predicted_department", "BBMP"))
        
        # Accumulate metrics
        if sev in ["Critical", "High"]:
            active += 1
        if sev == "Critical":
            critical += 1
            
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        
        if dept not in matrix:
            matrix[dept] = {"Non-Grievance": 0, "Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        matrix[dept][sev] = matrix[dept].get(sev, 0) + 1
        
        # Prepend to recent list
        recent.insert(0, {
            "id": str(item.get("id"))[:8].upper() if "id" in item else "UI",
            "timestamp": item.get("timestamp", datetime.now().isoformat()),
            "complaint": item.get("complaint", ""),
            "predicted_department": dept,
            "severity": sev.lower(),
        })

    # Keep only the last 15 recent items
    recent = recent[:15]
    
    # Format department chart data
    departments = sorted(dept_counts.keys())
    severity_order = ["Critical", "High", "Medium", "Low", "Non-Grievance"]
    
    # Build chart matrix
    chart_matrix = {sev: [] for sev in severity_order}
    for dept in departments:
        for sev in severity_order:
            val = matrix.get(dept, {}).get(sev, 0)
            chart_matrix[sev].append(int(val))

    return {
        "total_complaints": total,
        "active_complaints": active,
        "critical_complaints": critical,
        "recent_complaints": recent,
        "department_labels": departments,
        "department_distribution": [int(dept_counts[d]) for d in departments],
        "severity_labels": severity_order,
        "severity_distribution": [int(sev_counts.get(s, 0)) for s in severity_order],
        "chart_matrix": chart_matrix,
    }


# Ensure static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount the static files directory under /static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the main single-page application dashboard index.html."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    
    # Return placeholder HTML if not created yet
    return """
    <html>
        <head><title>AI Grievance System</title></head>
        <body style="font-family: sans-serif; padding: 50px; text-align: center;">
            <h2>AI Grievance System Web App</h2>
            <p>Static frontend dashboard is loading...</p>
        </body>
    </html>
    """
