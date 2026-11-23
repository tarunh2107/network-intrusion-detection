"""FastAPI application for network intrusion detection and security monitoring.

Phase 7 Requirements:
- Lifespan management with single model loading and checksum verification.
- Endpoints: GET /health, GET /model/info, POST /predict, POST /predict/batch, GET /alerts, GET /statistics.
- Strict input validation rejecting booleans, strings, non-finite values, and schema mismatches.
- Atomic SQLite persistence with rollback on error.
- Thread-safe CPU-bound inference preventing event-loop blockage.
- Clear error handling without exposing internal stack traces or private paths.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app.config import Settings, VerifiedModel, get_settings, load_verified_model
from app.ml.predict import (
    predict_batch_flows,
    predict_single_flow,
)
from app.services.storage import (
    check_database_readiness,
    get_alerts as storage_get_alerts,
    get_statistics as storage_get_statistics,
    init_database,
    record_prediction,
    record_predictions_batch,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -----------------------------------------------------------------------------

class SingleFlowRequest(BaseModel):
    """Payload for single network flow prediction."""
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping exactly 77 feature names to finite numeric values or null.",
    )


class BatchFlowItem(BaseModel):
    """Item within a batch request."""
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping exactly 77 feature names to finite numeric values or null.",
    )


class BatchFlowsRequest(BaseModel):
    """Payload for batch network flow prediction (1 to 500 flows)."""
    flows: List[BatchFlowItem] = Field(
        ...,
        description="List of flow objects containing features (1 to 500 items).",
    )


class PredictionResponse(BaseModel):
    """Prediction output for a single network flow."""
    prediction_id: str
    timestamp_utc: str
    prediction_label: str
    attack_probability: float
    decision_threshold: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Prediction output for batch request."""
    batch_size: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health and component readiness."""
    status: str
    service_ready: bool
    model_loaded: bool
    database_connected: bool


class ModelInfoResponse(BaseModel):
    """Metadata regarding loaded model artifact and evaluation contract."""
    model_version: str
    feature_count: int
    feature_names: List[str]
    classes: Dict[str, str]
    decision_threshold: float
    missing_value_policy: str
    offline_use_limitations: str


class AlertItem(BaseModel):
    """Stored ATTACK prediction record."""
    prediction_id: str
    timestamp_utc: str
    model_version: str
    decision_threshold: float
    attack_probability: float
    prediction_label: str


class AlertsResponse(BaseModel):
    """Paginated alerts query response."""
    total_alerts: int
    limit: int
    offset: int
    alerts: List[AlertItem]


class StatisticsResponse(BaseModel):
    """Historical classification summary metrics."""
    total_predictions: int
    benign_predictions: int
    attack_predictions: int
    attack_proportion: float
    disclaimer: str


# -----------------------------------------------------------------------------
# Lifespan Management
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and verify/load model once during app startup."""
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    app.state.model = None
    app.state.model_error = None

    # 1. Initialize SQLite Database
    try:
        init_database(settings.database_path)
    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)

    # 2. Verify and load model
    try:
        app.state.model = load_verified_model(settings)
    except Exception as exc:
        app.state.model_error = str(exc)
        logger.error("Failed to load verified model: %s", exc)

    yield

    # Clean shutdown
    app.state.model = None


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Application factory allowing custom settings for testing."""
    app = FastAPI(
        title="Network Intrusion Detection & Monitoring API",
        description="Production REST API for offline binary classification of network flows.",
        version="1.0.0",
        lifespan=lifespan,
    )

    if custom_settings:
        app.state.settings = custom_settings

    # -------------------------------------------------------------------------
    # Route Handlers
    # -------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health_check(response: Response) -> Dict[str, Any]:
        """Check system, model, and database readiness."""
        settings: Settings = getattr(app.state, "settings", get_settings())
        model: Optional[VerifiedModel] = getattr(app.state, "model", None)

        db_ready = check_database_readiness(settings.database_path)
        model_ready = model is not None

        is_healthy = db_ready and model_ready

        if not is_healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "unhealthy",
                "service_ready": False,
                "model_loaded": model_ready,
                "database_connected": db_ready,
            }

        return {
            "status": "healthy",
            "service_ready": True,
            "model_loaded": True,
            "database_connected": True,
        }

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> Dict[str, Any]:
        """Return model metadata, ordered features, class mapping, and threshold."""
        model: Optional[VerifiedModel] = getattr(app.state, "model", None)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model service is currently unavailable.",
            )

        return {
            "model_version": model.model_version,
            "feature_count": len(model.feature_names),
            "feature_names": model.feature_names,
            "classes": {"0": "BENIGN", "1": "ATTACK"},
            "decision_threshold": model.frozen_threshold,
            "missing_value_policy": (
                "Explicit JSON null values are imputed with training-set medians inside "
                "the Scikit-learn Pipeline. Negative rate features are sanitized to missing; "
                "negative TCP window sentinels are preserved. Entirely null vectors are rejected."
            ),
            "offline_use_limitations": (
                "Trained for offline flow record classification on CICIDS2017 benchmark data. "
                "Does not perform real-time packet capture, live sniffing, or temporal drift tracking."
            ),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: SingleFlowRequest) -> Dict[str, Any]:
        """Classify a single network flow and persist prediction history."""
        model: Optional[VerifiedModel] = getattr(app.state, "model", None)
        settings: Settings = getattr(app.state, "settings", get_settings())

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model service is currently unavailable.",
            )

        try:
            pred_res = predict_single_flow(req.features, model)
        except ValueError as val_err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(val_err),
            ) from val_err

        pred_id = str(uuid.uuid4())
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Atomic persistence
        try:
            record_prediction(
                db_path=settings.database_path,
                prediction_id=pred_id,
                timestamp_utc=timestamp_utc,
                model_version=pred_res.model_version,
                threshold=pred_res.decision_threshold,
                attack_probability=pred_res.attack_probability,
                prediction_label=pred_res.prediction_label,
            )
        except Exception as exc:
            logger.error("Failed persisting prediction %s: %s", pred_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Persistence error; prediction was not recorded.",
            ) from exc

        return {
            "prediction_id": pred_id,
            "timestamp_utc": timestamp_utc,
            "prediction_label": pred_res.prediction_label,
            "attack_probability": pred_res.attack_probability,
            "decision_threshold": pred_res.decision_threshold,
            "model_version": pred_res.model_version,
        }

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(req: BatchFlowsRequest) -> Dict[str, Any]:
        """Classify 1 to 500 network flows in a batch and persist history atomically."""
        model: Optional[VerifiedModel] = getattr(app.state, "model", None)
        settings: Settings = getattr(app.state, "settings", get_settings())

        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model service is currently unavailable.",
            )

        flows_raw = [item.features for item in req.flows]

        # 1. Validation and inference
        try:
            results = predict_batch_flows(flows_raw, model)
        except ValueError as val_err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(val_err),
            ) from val_err

        # 2. Prepare database records and response
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        records_to_insert = []
        response_items = []

        for res in results:
            pid = str(uuid.uuid4())
            records_to_insert.append((
                pid,
                timestamp_utc,
                res.model_version,
                res.decision_threshold,
                res.attack_probability,
                res.prediction_label,
            ))
            response_items.append({
                "prediction_id": pid,
                "timestamp_utc": timestamp_utc,
                "prediction_label": res.prediction_label,
                "attack_probability": res.attack_probability,
                "decision_threshold": res.decision_threshold,
                "model_version": res.model_version,
            })

        # 3. Atomic persistence
        try:
            record_predictions_batch(settings.database_path, records_to_insert)
        except Exception as exc:
            logger.error("Failed persisting batch predictions: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch persistence error; predictions were not recorded.",
            ) from exc

        return {
            "batch_size": len(response_items),
            "predictions": response_items,
        }

    @app.get("/alerts", response_model=AlertsResponse)
    def get_alerts(
        limit: int = Query(50, ge=1, le=500, description="Maximum number of alerts to return (1-500)."),
        offset: int = Query(0, ge=0, description="Pagination offset."),
        start_time: Optional[str] = Query(None, description="Optional start ISO 8601 timestamp."),
        end_time: Optional[str] = Query(None, description="Optional end ISO 8601 timestamp."),
    ) -> Dict[str, Any]:
        """Query stored ATTACK prediction alerts with pagination and date range filtering."""
        settings: Settings = getattr(app.state, "settings", get_settings())
        try:
            alerts_data = storage_get_alerts(
                db_path=settings.database_path,
                limit=limit,
                offset=offset,
                start_time=start_time,
                end_time=end_time,
            )
            return alerts_data
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err
        except Exception as exc:
            logger.error("Failed querying alerts: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error querying alerts history.",
            ) from exc

    @app.get("/statistics", response_model=StatisticsResponse)
    def get_statistics() -> Dict[str, Any]:
        """Retrieve aggregated historical classification metrics."""
        settings: Settings = getattr(app.state, "settings", get_settings())
        try:
            return storage_get_statistics(settings.database_path)
        except Exception as exc:
            logger.error("Failed querying statistics: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error querying prediction statistics.",
            ) from exc

    return app


# Default application instance for Uvicorn
app = create_app()
