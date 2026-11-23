"""Configuration settings for the Streamlit security-monitoring dashboard.

Phase 8 Requirements:
- Reads API_BASE_URL from configuration with default http://127.0.0.1:8000.
- Reads evaluation report and demo CSV paths.
- Provides central configuration for upload constraints and timeouts.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Optional

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _load_env_if_present() -> None:
    """Read local .env file into os.environ if present without overwriting."""
    env_file = WORKSPACE_ROOT / ".env"
    if env_file.is_file():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


_load_env_if_present()


@dataclasses.dataclass(frozen=True)
class DashboardSettings:
    """Dashboard runtime configuration."""

    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    evaluation_report_path: pathlib.Path = pathlib.Path(
        os.getenv(
            "EVALUATION_REPORT_PATH",
            str(WORKSPACE_ROOT / "reports" / "generated" / "test_evaluation.json"),
        )
    )
    demo_csv_path: pathlib.Path = pathlib.Path(
        os.getenv(
            "DEMO_CSV_PATH",
            str(WORKSPACE_ROOT / "data" / "demo" / "example_flows.csv"),
        )
    )
    demo_csv_1000_path: pathlib.Path = pathlib.Path(
        os.getenv(
            "DEMO_CSV_1000_PATH",
            str(WORKSPACE_ROOT / "data" / "demo" / "example_flows_1000.csv"),
        )
    )
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB cap
    max_upload_rows: int = 1000  # 1000 flow rows cap
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 15.0


def get_dashboard_settings() -> DashboardSettings:
    """Return dashboard configuration instance."""
    return DashboardSettings()
