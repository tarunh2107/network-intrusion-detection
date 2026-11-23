"""HTTP Client for communicating with the FastAPI intrusion-detection backend.

Phase 8 Requirements:
- Explicit connection and read timeouts.
- Graceful handling of unavailable API, HTTP 422, and HTTP 500/503 errors.
- Never logs or exposes uploaded raw feature payloads.
- Does not cache POST requests.
- On timeout, explains that the server may have recorded predictions; does NOT retry POSTs.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional

import requests

from app.dashboard.config import DashboardSettings, get_dashboard_settings

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ChunkedPredictionResult:
    """Encapsulates combined results from chunked batch submissions."""
    predictions: List[Dict[str, Any]]
    total_requested: int
    total_completed: int
    is_partial: bool
    error_message: Optional[str] = None
    chunks_sent: int = 0


class ApiClientError(Exception):
    """Base exception for API client operations."""


class ApiConnectionError(ApiClientError):
    """Raised when the backend API cannot be reached."""


class ApiTimeoutError(ApiClientError):
    """Raised when a request to the backend times out."""


class ApiValidationError(ApiClientError):
    """Raised when the backend rejects a payload with HTTP 422."""


class ApiServiceUnavailableError(ApiClientError):
    """Raised when the backend returns HTTP 503."""


class NidsApiClient:
    """REST client connecting the dashboard to the FastAPI backend."""

    def __init__(self, settings: Optional[DashboardSettings] = None, session: Optional[requests.Session] = None):
        self.settings = settings or get_dashboard_settings()
        self.base_url = self.settings.api_base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = (self.settings.connect_timeout_seconds, self.settings.read_timeout_seconds)

    def get_health(self) -> Dict[str, Any]:
        """Query GET /health for backend service readiness."""
        url = f"{self.base_url}/health"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 503:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                detail = data.get("detail", "Service dependencies unready.")
                raise ApiServiceUnavailableError(f"API unready (HTTP 503): {detail}")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as exc:
            raise ApiConnectionError(
                f"Cannot connect to API at {self.base_url}. Ensure the FastAPI server is running."
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            raise ApiTimeoutError(f"Health check read timed out after {self.settings.read_timeout_seconds}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Failed querying /health: {exc}") from exc

    def get_model_info(self) -> Dict[str, Any]:
        """Query GET /model/info for metadata and 77 feature names."""
        url = f"{self.base_url}/model/info"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 503:
                raise ApiServiceUnavailableError("Model service is unavailable on the backend (HTTP 503).")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as exc:
            raise ApiConnectionError(
                f"Cannot connect to API at {self.base_url}. Ensure the FastAPI server is running."
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            raise ApiTimeoutError(f"Model info read timed out after {self.settings.read_timeout_seconds}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Failed querying /model/info: {exc}") from exc

    def get_statistics(self) -> Dict[str, Any]:
        """Query GET /statistics for current prediction-history metrics."""
        url = f"{self.base_url}/statistics"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 503:
                raise ApiServiceUnavailableError("Statistics service unavailable (HTTP 503).")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as exc:
            raise ApiConnectionError(
                f"Cannot connect to API at {self.base_url}. Ensure the FastAPI server is running."
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            raise ApiTimeoutError(f"Statistics read timed out after {self.settings.read_timeout_seconds}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Failed querying /statistics: {exc}") from exc

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query GET /alerts with pagination and optional UTC date filters."""
        url = f"{self.base_url}/alerts"
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 400:
                detail = resp.json().get("detail", "Invalid query parameters.")
                raise ApiClientError(f"Invalid alert query (HTTP 400): {detail}")
            elif resp.status_code == 503:
                raise ApiServiceUnavailableError("Alerts service unavailable (HTTP 503).")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as exc:
            raise ApiConnectionError(
                f"Cannot connect to API at {self.base_url}. Ensure the FastAPI server is running."
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            raise ApiTimeoutError(f"Alerts query timed out after {self.settings.read_timeout_seconds}s.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Failed querying /alerts: {exc}") from exc

    def predict_batch(self, flows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit a single batch of up to 500 flows to POST /predict/batch for classification.

        IMPORTANT:
        - Strictly capped at 500 flows to match the FastAPI server schema.
        - Does NOT automatically retry on timeout or failure to prevent duplicate database rows.
        - Never logs or exposes raw flow vectors.
        """
        if not flows:
            raise ValueError("Flows list cannot be empty.")
        if len(flows) > 500:
            raise ValueError(
                f"Batch exceeds API server limit of 500 flows (got {len(flows)}). "
                "Use predict_in_chunks for larger datasets up to 1,000 flows."
            )

        url = f"{self.base_url}/predict/batch"
        payload = {"flows": flows}

        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 422:
                err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                detail = err_data.get("detail", "Input validation rejected by server.")
                raise ApiValidationError(f"Server validation error (HTTP 422): {detail}")
            elif resp.status_code == 503:
                raise ApiServiceUnavailableError("Inference service unavailable (HTTP 503).")
            elif resp.status_code == 500:
                raise ApiClientError("Server internal persistence failure (HTTP 500).")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as exc:
            raise ApiConnectionError(
                f"Cannot connect to API at {self.base_url}. Ensure the FastAPI server is running."
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            # Do NOT retry POST requests automatically!
            raise ApiTimeoutError(
                "Prediction request timed out. The server may have processed or recorded some predictions "
                "in the database; do not automatically retry to avoid creating duplicate records."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Prediction batch failed: {exc}") from exc

    def predict_in_chunks(
        self,
        flows: List[Dict[str, Any]],
        chunk_size: int = 500,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ChunkedPredictionResult:
        """Submit flows in sequential batches of up to chunk_size (default 500).

        Requirements:
        - Validates the entire upload first (capped at max_upload_rows = 1,000).
        - Dispatches sequential chunks of up to 500 flows matching the server contract.
        - Combines successful results in original input order.
        - If a chunk fails, stops immediately, preserves completed results so far,
          and marks the result as partial with an explicit warning.
        - NEVER automatically retries failed POST requests (avoids duplicate records).
        """
        if not flows:
            raise ValueError("Flows list cannot be empty.")
        if len(flows) > self.settings.max_upload_rows:
            raise ValueError(
                f"Batch exceeds maximum allowed size of {self.settings.max_upload_rows} flows (got {len(flows)})."
            )

        chunks = [flows[i : i + chunk_size] for i in range(0, len(flows), chunk_size)]
        combined_predictions: List[Dict[str, Any]] = []

        for chunk_idx, chunk in enumerate(chunks):
            try:
                resp = self.predict_batch(chunk)
                chunk_preds = resp.get("predictions", [])
                combined_predictions.extend(chunk_preds)
                if progress_callback:
                    progress_callback(len(combined_predictions), len(flows))
            except Exception as exc:
                err_msg = (
                    f"Batch {chunk_idx + 1} of {len(chunks)} failed after {len(combined_predictions)} of {len(flows)} flows completed. "
                    f"Reason: {exc}. "
                    "Notice: Do NOT automatically retry POST requests. A timeout or network error may have occurred "
                    "after the server recorded predictions in the database, and retrying will create duplicate records."
                )
                logger.error(err_msg)
                return ChunkedPredictionResult(
                    predictions=combined_predictions,
                    total_requested=len(flows),
                    total_completed=len(combined_predictions),
                    is_partial=True,
                    error_message=err_msg,
                    chunks_sent=chunk_idx + 1,
                )

        return ChunkedPredictionResult(
            predictions=combined_predictions,
            total_requested=len(flows),
            total_completed=len(combined_predictions),
            is_partial=False,
            error_message=None,
            chunks_sent=len(chunks),
        )
