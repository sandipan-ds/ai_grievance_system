from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from src.db.postgres import create_postgres_engine, test_connection
from src.logger.logger import append_prediction_log
from src.ml.model_loader import ModelBundle, load_model_bundle
from src.ml.predictor import predict_complaint
from src.schemas.schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_model_bundle()

    try:
        app.state.db_engine = create_postgres_engine()
        app.state.db_connected = test_connection(app.state.db_engine)
    except Exception:
        app.state.db_engine = None
        app.state.db_connected = False

    yield


app = FastAPI(
    title="AI Grievance System API",
    version="v1.0",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    try:
        models: ModelBundle = request.app.state.models
        predicted_department, severity = predict_complaint(models, payload.complaint)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    background_tasks.add_task(
        append_prediction_log,
        complaint=payload.complaint,
        predicted_department=predicted_department,
        severity=severity,
        model_version="v1.0",
    )

    return PredictionResponse(
        predicted_department=predicted_department,
        severity=severity,
    )
