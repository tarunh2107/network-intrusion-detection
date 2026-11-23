"""Network Intrusion Monitor - Streamlit Security Monitoring Dashboard.

Phase 8 Requirements:
- Connects exclusively to the FastAPI backend REST API.
- Offline network-flow classification.
- Clean visual design with Green (BENIGN), Red (ATTACK), and distinct disconnected states.
- 4 navigation views: Overview, Analyze Flows, Alerts, Model & Evaluation.
- Strict CSV validation, explicit analysis submission, and session state persistence.
- Read-only held-out benchmark evaluation with Bot weakness and rare-category callouts.
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiServiceUnavailableError,
    ApiTimeoutError,
    ApiValidationError,
    ChunkedPredictionResult,
    NidsApiClient,
)
from app.dashboard.config import DashboardSettings, get_dashboard_settings
from app.dashboard.csv_utils import generate_header_template_csv, inspect_and_validate_csv

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Streamlit App Setup & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Network Intrusion Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished typography, metric cards, and badge styling
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #1e293b;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .badge-benign {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-attack {
        background-color: #f8d7da;
        color: #842029;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-offline {
        background-color: #e2e8f0;
        color: #475569;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-card-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .notice-box {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 0.8rem 1rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
        color: #334155;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client() -> NidsApiClient:
    """Return cached NidsApiClient instance."""
    return NidsApiClient()


def check_backend_status(client: NidsApiClient) -> Tuple[bool, str, Dict[str, Any]]:
    """Check backend health without stale caching."""
    try:
        health = client.get_health()
        return True, "API Connected", health
    except ApiConnectionError as conn_err:
        return False, "API Disconnected", {"error": str(conn_err)}
    except ApiServiceUnavailableError as unready_err:
        return False, "API Unready", {"error": str(unready_err)}
    except Exception as exc:
        return False, "Connection Error", {"error": str(exc)}


# -----------------------------------------------------------------------------
# Navigation & Sidebar
# -----------------------------------------------------------------------------

client = get_client()
settings = client.settings

# Check backend status dynamically on each run
is_connected, status_text, health_info = check_backend_status(client)

with st.sidebar:
    st.markdown("### 🛡️ Intrusion Monitor")
    st.markdown(
        f"<span class='badge-offline'>Backend: {settings.api_base_url}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # System Status Pill
    if is_connected:
        st.success(f"● {status_text}")
    else:
        st.error(f"● {status_text}")
        st.caption("Ensure FastAPI is running on the configured host & port.")

    st.markdown("---")
    view_selection = st.radio(
        "Navigation",
        options=["Overview", "Analyze Flows", "Alerts", "Model & Evaluation"],
        index=0,
    )

    st.markdown("---")
    st.caption("Offline Network Intrusion Detection System")
    st.caption("Benchmark: CICIDS2017 flow collection")


# Header
st.markdown("<div class='main-title'>Network Intrusion Monitor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Offline network-flow classification & security monitoring</div>",
    unsafe_allow_html=True,
)

# If backend is disconnected, display top notification
if not is_connected:
    st.warning(
        f"⚠️ **Backend Unavailable**: Cannot connect to the FastAPI backend at `{settings.api_base_url}`. "
        "Predictions, alerts, and model queries are disabled. "
        "Please start the backend service using:\n\n"
        "```powershell\n.\\.venv\\Scripts\\uvicorn.exe app.api.main:app --host 127.0.0.1 --port 8000\n```"
    )

# -----------------------------------------------------------------------------
# View 1: Overview
# -----------------------------------------------------------------------------

if view_selection == "Overview":
    col_hdr, col_btn = st.columns([4, 1])
    with col_hdr:
        st.markdown("### System Overview & Traffic Activity")
    with col_btn:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    if not is_connected:
        st.info("System metrics will load once the backend API service is connected.")
    else:
        # Fetch statistics and model info
        stats: Optional[Dict[str, Any]] = None
        model_info: Optional[Dict[str, Any]] = None
        recent_alerts: Optional[Dict[str, Any]] = None
        error_msg: Optional[str] = None

        try:
            stats = client.get_statistics()
            model_info = client.get_model_info()
            recent_alerts = client.get_alerts(limit=5, offset=0)
        except ApiClientError as err:
            error_msg = str(err)

        if error_msg:
            st.error(f"Failed loading system statistics: {error_msg}")
        elif stats and model_info:
            # Top Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""<div class='metric-card'>
                        <div class='metric-card-label'>Total Predictions</div>
                        <div class='metric-card-value'>{stats['total_predictions']:,}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class='metric-card'>
                        <div class='metric-card-label'>Benign Flows</div>
                        <div class='metric-card-value' style='color:#0f5132;'>{stats['benign_predictions']:,}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""<div class='metric-card'>
                        <div class='metric-card-label'>Attack Flows</div>
                        <div class='metric-card-value' style='color:#842029;'>{stats['attack_predictions']:,}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                prop_str = f"{stats['attack_proportion'] * 100:.2f}%"
                st.markdown(
                    f"""<div class='metric-card'>
                        <div class='metric-card-label'>Attack Proportion</div>
                        <div class='metric-card-value'>{prop_str}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Metadata info bar
            st.markdown(
                f"**Active Model:** `{model_info.get('model_version', 'baseline_v1')}` | "
                f"**Decision Threshold:** `{model_info.get('decision_threshold', 0.29):.2f}` | "
                f"**Features Evaluated:** `{model_info.get('feature_count', 77)}`"
            )

            # Empty state vs Charts
            if stats["total_predictions"] == 0:
                st.info(
                    "ℹ️ **No network flows have been analyzed yet.** "
                    "Upload and analyze flow CSV records in the **Analyze Flows** tab to generate predictions and security alerts."
                )
            else:
                col_chart, col_alerts = st.columns([1, 1])

                with col_chart:
                    st.markdown("#### Classification Breakdown")
                    chart_df = pd.DataFrame(
                        {
                            "Classification": ["BENIGN", "ATTACK"],
                            "Count": [stats["benign_predictions"], stats["attack_predictions"]],
                        }
                    )
                    fig = px.bar(
                        chart_df,
                        x="Classification",
                        y="Count",
                        color="Classification",
                        color_discrete_map={"BENIGN": "#198754", "ATTACK": "#dc3545"},
                        text="Count",
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=280,
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis_title="Flow Records",
                        xaxis_title=None,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_alerts:
                    st.markdown("#### Recent Security Alerts")
                    alerts_list = recent_alerts.get("alerts", []) if recent_alerts else []
                    if not alerts_list:
                        st.info("No attack detections recorded.")
                    else:
                        table_data = []
                        for a in alerts_list:
                            table_data.append(
                                {
                                    "ID": a["prediction_id"][:8] + "...",
                                    "Timestamp (UTC)": a["timestamp_utc"],
                                    "Score": f"{a['attack_probability']:.4f}",
                                    "Label": a["prediction_label"],
                                }
                            )
                        df_recent = pd.DataFrame(table_data)
                        st.dataframe(df_recent, use_container_width=True, hide_index=True)

            # Disclaimer
            st.markdown(
                """
                <div class='notice-box'>
                    ⚠️ <strong>Operational Scope Notice</strong>:
                    All statistics represent statistical model predictions on extracted network-flow records,
                    not verified physical security incidents. Timestamps are formatted in UTC.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# View 2: Analyze Flows
# -----------------------------------------------------------------------------

elif view_selection == "Analyze Flows":
    st.markdown("### Offline Network Flow Analysis")
    st.markdown(
        "Upload a batch of network flow feature vectors (CSV format) for classification against the frozen pipeline."
    )

    if not is_connected:
        st.error("Inference is unavailable while the backend API is disconnected.")
    else:
        # Fetch expected 77 features from model info
        model_info = None
        try:
            model_info = client.get_model_info()
            expected_features = model_info.get("feature_names", [])
        except Exception as exc:
            st.error(f"Failed fetching feature schema from backend: {exc}")
            expected_features = []

        if expected_features:
            # Downloadable templates
            col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
            with col_t1:
                header_csv = generate_header_template_csv(expected_features)
                st.download_button(
                    label="📥 Download 77-Feature Header Template (CSV)",
                    data=header_csv,
                    file_name="flow_schema_headers_template.csv",
                    mime="text/csv",
                    help="CSV containing exactly the 77 required model features.",
                )
            with col_t2:
                demo_path = settings.demo_csv_path
                if demo_path.is_file():
                    with open(demo_path, "r", encoding="utf-8") as f:
                        demo_csv_data = f.read()
                    st.download_button(
                        label="📥 Download Benchmark Example CSV (10 flows)",
                        data=demo_csv_data,
                        file_name="benchmark_example_flows.csv",
                        mime="text/csv",
                        help="Pre-extracted benchmark training sample with exactly 77 feature columns.",
                    )
                else:
                    st.caption("Benchmark example CSV is currently unavailable.")
            with col_t3:
                demo_1000_path = settings.demo_csv_1000_path
                if demo_1000_path.is_file():
                    with open(demo_1000_path, "r", encoding="utf-8") as f:
                        demo_1000_csv_data = f.read()
                    st.download_button(
                        label="📥 Download Benchmark Example CSV (1,000 flows)",
                        data=demo_1000_csv_data,
                        file_name="benchmark_example_flows_1000.csv",
                        mime="text/csv",
                        help="Pre-extracted benchmark training sample with 1,000 flows across 77 feature columns.",
                    )
                else:
                    st.caption("1,000-flow benchmark example CSV is currently unavailable.")

            st.markdown("---")

            # Upload constraints notice
            st.caption(
                f"**Upload Constraints:** Capped at {settings.max_upload_size_bytes // (1024*1024)} MB and "
                f"{settings.max_upload_rows} data rows. Must contain exactly the 77 named feature columns. "
                "Blank cells are imputed with training medians. Non-numeric values and infinities are strictly rejected."
            )

            uploaded_file = st.file_uploader(
                "Upload Network Flow Records (CSV)",
                type=["csv"],
                help=f"Select a CSV file containing up to {settings.max_upload_rows} network flows.",
            )

            if uploaded_file is not None:
                # Read bytes and validate
                file_bytes = uploaded_file.getvalue()
                flows_payload, preview_df, validation_errors = inspect_and_validate_csv(
                    file_bytes_or_text=file_bytes,
                    expected_feature_names=expected_features,
                    max_rows=settings.max_upload_rows,
                    max_size_bytes=settings.max_upload_size_bytes,
                )

                if validation_errors:
                    st.error(
                        f"❌ **CSV Validation Failed ({len(validation_errors)} issue{'s' if len(validation_errors) > 1 else ''}):**"
                    )
                    for err in validation_errors:
                        st.markdown(f"- {err}")
                    st.info("Please correct the CSV schema errors above and re-upload.")
                else:
                    st.success(
                        f"✅ **CSV Conforms to Contract**: {len(flows_payload)} flow record(s) validated and ready for classification."
                    )

                    with st.expander("Preview Uploaded Data (First 10 Rows)", expanded=True):
                        st.dataframe(preview_df.head(10), use_container_width=True)

                    # Submit button
                    col_act1, col_act2 = st.columns([2, 3])
                    with col_act1:
                        analyze_clicked = st.button(
                            "🚀 Analyze Flows",
                            type="primary",
                            disabled=st.session_state.get("analyzing_in_progress", False),
                            use_container_width=True,
                        )

                    if analyze_clicked:
                        st.session_state["analyzing_in_progress"] = True
                        progress_bar = st.progress(0, text=f"Preparing {len(flows_payload)} flows for batch scoring...")

                        def on_progress(completed: int, total: int) -> None:
                            pct = min(1.0, completed / total)
                            progress_bar.progress(pct, text=f"Scored {completed}/{total} flows across sequential batches...")

                        try:
                            with st.spinner(f"Scoring {len(flows_payload)} flow(s) in sequential batches..."):
                                chunked_result = client.predict_in_chunks(
                                    flows_payload,
                                    chunk_size=500,
                                    progress_callback=on_progress,
                                )
                                st.session_state["last_analysis_results"] = dataclasses.asdict(chunked_result)
                                st.session_state["last_analyzed_row_count"] = len(flows_payload)
                                progress_bar.progress(
                                    1.0,
                                    text=f"Scoring complete: {chunked_result.total_completed}/{chunked_result.total_requested} flows."
                                )
                                if chunked_result.is_partial:
                                    st.warning(
                                        f"⚠️ Batch analysis partially completed: {chunked_result.total_completed} of {chunked_result.total_requested} flows processed."
                                    )
                                else:
                                    st.toast(f"Successfully scored {chunked_result.total_completed} flows!", icon="✅")
                        except ApiTimeoutError as t_err:
                            st.warning(f"⚠️ {t_err}")
                        except ApiValidationError as val_err:
                            st.error(f"❌ {val_err}")
                        except ApiClientError as c_err:
                            st.error(f"❌ {c_err}")
                        except Exception as exc:
                            st.error(f"❌ Unexpected error during batch analysis: {exc}")
                        finally:
                            st.session_state["analyzing_in_progress"] = False

            # Display Analysis Results from session state (preserves across tab switches)
            if "last_analysis_results" in st.session_state:
                st.markdown("---")
                st.markdown("### Analysis Results")
                pred_data = st.session_state["last_analysis_results"]
                predictions = pred_data.get("predictions", [])

                if pred_data.get("is_partial", False):
                    st.warning(
                        f"⚠️ **Partial Results Returned ({pred_data.get('total_completed', 0)} of {pred_data.get('total_requested', 0)} flows completed):**\n\n"
                        f"{pred_data.get('error_message')}\n\n"
                        "**Important:** Do NOT automatically retry POST requests. Completed batches were already committed to the database, and retrying will create duplicate entries."
                    )

                if predictions:
                    benign_cnt = sum(1 for p in predictions if p.get("prediction_label") == "BENIGN")
                    attack_cnt = sum(1 for p in predictions if p.get("prediction_label") == "ATTACK")

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        if pred_data.get("is_partial", False):
                            st.metric("Flows Completed", f"{len(predictions)} / {pred_data.get('total_requested', len(predictions))}")
                        else:
                            st.metric("Total Flows Analyzed", len(predictions))
                    with r2:
                        st.metric("Classified BENIGN", benign_cnt)
                    with r3:
                        st.metric("Classified ATTACK", attack_cnt)

                    if pred_data.get("chunks_sent", 1) > 1:
                        st.caption(f"ℹ️ Dispatched across {pred_data.get('chunks_sent')} sequential batches (up to 500 flows each).")

                    # Build display table preserving input order
                    display_rows = []
                    for idx, p in enumerate(predictions, start=1):
                        display_rows.append(
                            {
                                "#": idx,
                                "Prediction ID": p["prediction_id"],
                                "Classification": p["prediction_label"],
                                "Attack Probability": f"{p['attack_probability']:.4f}",
                                "Threshold": f"{p['decision_threshold']:.2f}",
                                "Timestamp (UTC)": p["timestamp_utc"],
                            }
                        )

                    df_results = pd.DataFrame(display_rows)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)

                    # Disclaimer on uncalibrated score
                    st.markdown(
                        """
                        <div class='notice-box'>
                            ℹ️ <strong>Score Interpretation Notice</strong>:
                            The attack score is an uncalibrated probability output from the Random Forest model.
                            It reflects statistical resemblance to known benchmark attack flows and does
                            <strong>not</strong> indicate physical severity, exploitability, or incident impact.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Download results CSV
                    csv_export_buffer = io.StringIO()
                    df_results.to_csv(csv_export_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_export_buffer.getvalue(),
                        file_name=f"flow_predictions_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )


# -----------------------------------------------------------------------------
# View 3: Alerts
# -----------------------------------------------------------------------------

elif view_selection == "Alerts":
    st.markdown("### Security Alert Log")
    st.markdown(
        "Historical log of network flow records classified as **ATTACK** by the inference pipeline."
    )

    if not is_connected:
        st.error("Alert logs are unavailable while the backend API is disconnected.")
    else:
        # Filters
        c_filter1, c_filter2, c_filter3 = st.columns([1, 1, 1])
        with c_filter1:
            limit = st.selectbox("Page Size", options=[10, 25, 50, 100], index=1)
        with c_filter2:
            start_date = st.date_input("Start Date (UTC, optional)", value=None)
        with c_filter3:
            end_date = st.date_input("End Date (UTC, optional)", value=None)

        # Pagination controls
        if "alerts_offset" not in st.session_state:
            st.session_state["alerts_offset"] = 0

        # Construct ISO timestamp filters if specified
        start_iso: Optional[str] = None
        end_iso: Optional[str] = None
        if start_date:
            start_iso = datetime.datetime.combine(start_date, datetime.time.min, tzinfo=datetime.timezone.utc).isoformat()
        if end_date:
            end_iso = datetime.datetime.combine(end_date, datetime.time.max, tzinfo=datetime.timezone.utc).isoformat()

        # Query /alerts
        alerts_response: Optional[Dict[str, Any]] = None
        alert_error: Optional[str] = None

        try:
            alerts_response = client.get_alerts(
                limit=limit,
                offset=st.session_state["alerts_offset"],
                start_time=start_iso,
                end_time=end_iso,
            )
        except ApiClientError as err:
            alert_error = str(err)

        if alert_error:
            st.error(f"Failed querying alerts: {alert_error}")
        elif alerts_response:
            total_alerts = alerts_response.get("total_alerts", 0)
            alerts_data = alerts_response.get("alerts", [])

            st.markdown(f"**Total Matching Alerts:** `{total_alerts}`")

            if not alerts_data:
                st.info("No security alerts found matching query filters.")
            else:
                table_rows = []
                for a in alerts_data:
                    table_rows.append(
                        {
                            "Prediction ID": a["prediction_id"],
                            "Timestamp (UTC)": a["timestamp_utc"],
                            "Label": a["prediction_label"],
                            "Attack Probability": f"{a['attack_probability']:.4f}",
                            "Threshold": f"{a['decision_threshold']:.2f}",
                            "Model Version": a["model_version"],
                        }
                    )
                df_alerts = pd.DataFrame(table_rows)
                st.dataframe(df_alerts, use_container_width=True, hide_index=True)

                # Pagination Buttons
                col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
                with col_p1:
                    if st.session_state["alerts_offset"] > 0:
                        if st.button("⬅️ Previous Page"):
                            st.session_state["alerts_offset"] = max(0, st.session_state["alerts_offset"] - limit)
                            st.rerun()
                with col_p2:
                    current_page = (st.session_state["alerts_offset"] // limit) + 1
                    max_page = max(1, (total_alerts + limit - 1) // limit)
                    st.caption(f"Page {current_page} of {max_page}")
                with col_p3:
                    if st.session_state["alerts_offset"] + limit < total_alerts:
                        if st.button("Next Page ➡️"):
                            st.session_state["alerts_offset"] += limit
                            st.rerun()

                # Export current page button
                buf = io.StringIO()
                df_alerts.to_csv(buf, index=False)
                st.download_button(
                    label="📥 Export Current Alerts Page (CSV)",
                    data=buf.getvalue(),
                    file_name=f"security_alerts_page_{current_page}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            # Disclaimer
            st.markdown(
                """
                <div class='notice-box'>
                    ℹ️ <strong>Alert Log Data Boundaries</strong>:
                    Stored alert records contain prediction IDs, timestamps, probability scores, and model versions.
                    Attack subcategories, severity ratings, IP addresses, and physical hostnames are
                    <strong>not</strong> stored in the prediction logs.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# View 4: Model & Evaluation
# -----------------------------------------------------------------------------

elif view_selection == "Model & Evaluation":
    st.markdown("### Model Architecture & Held-Out Benchmark Evaluation")
    st.markdown(
        "Inspect active model parameters and independently verified held-out benchmark performance artifacts."
    )

    # 1. Active Loaded Model Metadata
    st.markdown("#### 1. Loaded Model Metadata")
    if is_connected:
        try:
            m_info = client.get_model_info()
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Model Version", m_info.get("model_version", "baseline_v1"))
            with col_m2:
                st.metric("Features", m_info.get("feature_count", 77))
            with col_m3:
                st.metric("Frozen Threshold", f"{m_info.get('decision_threshold', 0.29):.2f}")
            with col_m4:
                st.metric("Target Classes", "0: BENIGN, 1: ATTACK")

            st.caption(f"**Missing Value Policy:** {m_info.get('missing_value_policy')}")
            st.caption(f"**Operational Scope:** {m_info.get('offline_use_limitations')}")
        except Exception as exc:
            st.warning(f"Could not retrieve model metadata from API: {exc}")
    else:
        st.info("API disconnected; model metadata cannot be inspected.")

    st.markdown("---")

    # 2. Held-Out Benchmark Evaluation (Phase 6B)
    st.markdown("#### 2. Held-Out Test Set Benchmark Evaluation (Phase 6B)")
    st.caption(
        "Read-only evaluation metrics scored exactly once on the frozen held-out test partition (416,144 flows). "
        "Evaluation is never rerun during inference."
    )

    report_path = settings.evaluation_report_path
    if not report_path.is_file():
        st.warning(f"Evaluation artifact report not found at `{report_path}`.")
    else:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception as exc:
            st.error(f"Failed loading evaluation report JSON: {exc}")
            eval_data = None

        if eval_data:
            frozen_cfg = eval_data.get("frozen_configuration", {})
            metrics = eval_data.get("metrics_at_frozen_threshold", {})
            cm = metrics.get("confusion_matrix", {})
            sample_counts = metrics.get("sample_counts", {})

            # Aggregate Metric Cards
            em1, em2, em3, em4, em5 = st.columns(5)
            with em1:
                st.metric("ATTACK Precision", f"{metrics.get('precision', 0.0) * 100:.2f}%")
            with em2:
                st.metric("ATTACK Recall", f"{metrics.get('recall', 0.0) * 100:.2f}%")
            with em3:
                st.metric("ATTACK F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")
            with em4:
                st.metric("False Positive Rate", f"{metrics.get('fpr', 0.0) * 100:.4f}%")
            with em5:
                st.metric("Average Precision", f"{metrics.get('average_precision', 0.0):.4f}")

            # Confusion Matrix Table
            st.markdown("##### Held-Out Confusion Matrix")
            col_cm1, col_cm2 = st.columns([1, 1])
            with col_cm1:
                cm_df = pd.DataFrame(
                    [
                        {"Actual": "BENIGN", "Predicted BENIGN (TN)": f"{cm.get('tn', 0):,}", "Predicted ATTACK (FP)": f"{cm.get('fp', 0):,}"},
                        {"Actual": "ATTACK", "Predicted BENIGN (FN)": f"{cm.get('fn', 0):,}", "Predicted ATTACK (TP)": f"{cm.get('tp', 0):,}"},
                    ]
                )
                st.dataframe(cm_df, use_container_width=True, hide_index=True)

            with col_cm2:
                st.markdown(
                    f"""
                    - **Total Evaluated Flows:** `{sample_counts.get('total', 416144):,}`
                    - **Benign Flows:** `{sample_counts.get('benign', 339256):,}` (81.52%)
                    - **Attack Flows:** `{sample_counts.get('attack', 76888):,}` (18.48%)
                    - **False Alarms (FP):** `{cm.get('fp', 640):,}` out of `{sample_counts.get('benign', 339256):,}` benign flows
                    - **Missed Attacks (FN):** `{cm.get('fn', 207):,}` out of `{sample_counts.get('attack', 76888):,}` attack flows
                    """
                )

            # Per-Class Recall Breakdown
            st.markdown("##### Per-Category Recall & Support Breakdown")
            per_class = metrics.get("per_class_recall", {})
            if per_class:
                class_rows = []
                for cat_name, cat_stats in per_class.items():
                    class_rows.append(
                        {
                            "Category": cat_name,
                            "Type": "ATTACK" if cat_stats.get("is_attack") else "BENIGN",
                            "Test Flows": cat_stats.get("total", 0),
                            "Detected Correct": cat_stats.get("detected_correct", 0),
                            "Missed": cat_stats.get("missed", 0),
                            "Recall": f"{cat_stats.get('recall', 0.0) * 100:.2f}%",
                            "Uncertainty Warning": cat_stats.get("uncertainty_warning") or "None",
                        }
                    )
                df_cat = pd.DataFrame(class_rows)
                st.dataframe(df_cat, use_container_width=True, hide_index=True)

            # Critical Methodological Warnings
            st.markdown(
                """
                <div class='notice-box' style='border-left-color: #f59e0b; background-color: #fffbeb;'>
                    ⚠️ <strong>Observed Model Weaknesses & Uncertainties</strong>:<br>
                    • <strong>Bot Detection Weakness</strong>: Bot recall is <strong>59.52%</strong> on the held-out test partition
                    (172 of 289 detected; 117 missed) and was 37.14% on the validation sample. This is an observed weakness of the baseline Random Forest model.<br>
                    • <strong>Extreme Variance on Rare Categories</strong>: Categories with minimal representation in the benchmark dataset
                    (e.g., <em>Heartbleed</em> with 1 test flow, <em>Web Attack - Sql Injection</em> with 2 flows) exhibit extreme estimation variance.
                    Metrics for these categories do not establish reliable generalizability.
                </div>
                """,
                unsafe_allow_html=True,
            )
