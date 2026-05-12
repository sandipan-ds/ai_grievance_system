from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# Avoid shadowing the Streamlit package with this file name.
SCRIPT_DIR = Path(__file__).resolve().parent
_clean_path = []
for entry in sys.path:
    try:
        resolved = Path(entry or ".").resolve()
    except Exception:
        _clean_path.append(entry)
        continue
    if resolved != SCRIPT_DIR:
        _clean_path.append(entry)
sys.path = _clean_path

import streamlit as st


PROJECT_ROOT = SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
REGISTRATION_LOG_PATH = LOG_DIR / "streamlit_registered_complaints.jsonl"
DEFAULT_BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")
API_TIMEOUT_SECONDS = 20
BACKEND_STARTUP_TIMEOUT_SECONDS = 30
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
SEVERITY_COLORS = {
    "Critical": "#b42318",
    "High": "#f97316",
    "Medium": "#f59e0b",
    "Low": "#0f766e",
    "Unknown": "#64748b",
}
DEPARTMENT_CANONICAL_MAP = {
    "Bruhat Bengaluru Mahanagara Palike": "BBMP",
    "Bangalore Water Supply And Sewerage Board": "BWSSB",
    "Bangalore Electricity Supply Company": "BESCOM",
    "Karnataka State Pollution Control Board": "KSPCB",
    "Bangalore Traffic Police": "Traffic Police",
    "BTP": "Traffic Police",
    "BCP": "Police Department",
    "BMTC": "Transport",
    "KSRTC": "Transport",
    "BDA": "BBMP",
    "KSFES": "Fire Services",
}


def normalize_backend_url(raw_url: str) -> str:
    return raw_url.strip().rstrip("/")


def build_prediction_url(base_url: str) -> str:
    return f"{normalize_backend_url(base_url)}/predict"


def build_openapi_url(base_url: str) -> str:
    return f"{normalize_backend_url(base_url)}/openapi.json"


def parse_backend_target(base_url: str) -> tuple[str, int]:
    parsed = urlparse(normalize_backend_url(base_url))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return host, port


def is_local_backend_target(base_url: str) -> bool:
    host, _ = parse_backend_target(base_url)
    return host in {"127.0.0.1", "localhost", "::1"}


def normalize_department(value: object) -> str:
    if pd.isna(value):
        return "Unassigned"
    text = str(value).strip()
    if not text:
        return "Unassigned"
    return DEPARTMENT_CANONICAL_MAP.get(text, text)


def normalize_severity(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    mapping = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    return mapping.get(text, "Unknown")


def truncate_text(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def parse_datetime_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


@st.cache_data(show_spinner=False)
def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "augmented_combined.csv",
        usecols=["description", "civic_agency_title", "severity", "created_at"],
    )
    return pd.DataFrame(
        {
            "case_id": "",
            "description": df["description"].fillna("").astype(str),
            "civic_agency_title": df["civic_agency_title"].apply(normalize_department),
            "severity": df["severity"].apply(normalize_severity),
            "created_at": parse_datetime_column(df["created_at"]),
            "source": "historical_dataset",
        }
    )


def load_registered_complaints() -> pd.DataFrame:
    columns = ["case_id", "description", "civic_agency_title", "severity", "created_at", "source"]
    if not REGISTRATION_LOG_PATH.exists():
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    with REGISTRATION_LOG_PATH.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                {
                    "case_id": str(payload.get("case_id", "")),
                    "description": str(payload.get("description", "")),
                    "civic_agency_title": normalize_department(payload.get("civic_agency_title")),
                    "severity": normalize_severity(payload.get("severity")),
                    "created_at": pd.to_datetime(payload.get("created_at"), errors="coerce", utc=True),
                    "source": "registered_via_ui",
                }
            )

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)


def prepare_operational_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["description"] = df["description"].fillna("").astype(str)
    df["civic_agency_title"] = df["civic_agency_title"].apply(normalize_department)
    df["severity"] = df["severity"].apply(normalize_severity)
    df["created_at"] = parse_datetime_column(df["created_at"])
    local_time = df["created_at"].dt.tz_convert("Asia/Calcutta")
    df["complaint_date"] = local_time.dt.date
    df["year"] = local_time.dt.year
    return df


def initialize_dashboard_state() -> None:
    if "dashboard_df" in st.session_state:
        return

    merged = pd.concat([load_base_dataset(), load_registered_complaints()], ignore_index=True)
    st.session_state.dashboard_df = prepare_operational_dataset(merged)
    st.session_state.latest_registration = None


def api_health_check(base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(build_openapi_url(base_url), timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        return False, f"FastAPI backend is not reachable: {exc}"
    return True, "FastAPI backend is reachable."


def start_local_backend(base_url: str) -> subprocess.Popen[bytes]:
    host, port = parse_backend_target(base_url)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_backend_running(base_url: str) -> tuple[bool, str]:
    ok, message = api_health_check(base_url)
    if ok:
        return True, message

    if not is_local_backend_target(base_url):
        return False, message

    process = st.session_state.get("backend_process")
    if process is not None and process.poll() is None:
        st.session_state["backend_launch_status"] = "Backend is starting..."
    else:
        try:
            st.session_state["backend_process"] = start_local_backend(base_url)
            st.session_state["backend_launch_status"] = "Backend startup requested."
        except FileNotFoundError as exc:
            return False, f"Could not start uvicorn: {exc}"
        except Exception as exc:
            return False, f"Could not start FastAPI backend: {exc}"

    deadline = perf_counter() + BACKEND_STARTUP_TIMEOUT_SECONDS
    while perf_counter() < deadline:
        ok, message = api_health_check(base_url)
        if ok:
            st.session_state["backend_launch_status"] = "Backend is ready."
            return True, message
        if process is not None and process.poll() is not None:
            break
        sleep(0.5)

    if process is not None and process.poll() is not None:
        return False, "FastAPI backend exited before it became ready. Check the terminal for uvicorn errors."
    return False, "FastAPI backend did not become ready in time."


def call_prediction_api(base_url: str, complaint: str) -> dict[str, str | float]:
    start = perf_counter()
    response = requests.post(
        build_prediction_url(base_url),
        json={"complaint": complaint},
        timeout=API_TIMEOUT_SECONDS,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = "Prediction request failed."
        try:
            error_payload = response.json()
            if isinstance(error_payload, dict) and error_payload.get("detail"):
                detail = str(error_payload["detail"])
        except Exception:
            if response.text:
                detail = response.text.strip()
        raise RuntimeError(detail) from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected response from FastAPI backend.")

    department = payload.get("predicted_department")
    severity = payload.get("severity")
    if not department or not severity:
        raise RuntimeError("FastAPI backend returned an incomplete prediction.")

    return {
        "predicted_department": normalize_department(department),
        "severity": normalize_severity(severity),
        "elapsed_seconds": perf_counter() - start,
    }


def append_registration_log(record: dict[str, str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRATION_LOG_PATH.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def register_new_complaint(complaint: str, prediction: dict[str, str | float]) -> dict[str, str]:
    record = {
        "case_id": str(uuid4())[:8].upper(),
        "description": complaint.strip(),
        "civic_agency_title": str(prediction["predicted_department"]),
        "severity": normalize_severity(prediction["severity"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "registered_via_ui",
    }
    append_registration_log(record)

    current = st.session_state.dashboard_df.drop(columns=["complaint_date", "year"])
    merged = pd.concat([current, pd.DataFrame([record])], ignore_index=True)
    st.session_state.dashboard_df = prepare_operational_dataset(merged)
    st.session_state.latest_registration = record
    return record


def build_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["civic_agency_title", "severity"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SEVERITY_ORDER, fill_value=0)
    )
    grouped["Total Complaints"] = grouped.sum(axis=1)
    grouped = grouped.reset_index().rename(columns={"civic_agency_title": "Civic Agency"})
    return grouped.sort_values("Total Complaints", ascending=False).reset_index(drop=True)


def build_recent_seven_day_window(df: pd.DataFrame) -> tuple[pd.DataFrame, list[object]]:
    dated_df = df.dropna(subset=["complaint_date"]).copy()
    unique_dates = sorted(dated_df["complaint_date"].dropna().unique())
    recent_dates = unique_dates[-7:]
    if not recent_dates:
        return dated_df.iloc[0:0].copy(), []
    return dated_df[dated_df["complaint_date"].isin(recent_dates)].copy(), recent_dates


def build_department_bar(summary_df: pd.DataFrame) -> go.Figure:
    top_df = summary_df.head(10).copy()
    fig = go.Figure()
    for severity in SEVERITY_ORDER:
        fig.add_bar(
            x=top_df["Civic Agency"],
            y=top_df[severity],
            name=severity,
            marker_color=SEVERITY_COLORS[severity],
        )
    fig.update_layout(
        barmode="stack",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title="Severity",
        xaxis_title="Civic Agency",
        yaxis_title="Complaint Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_department_donut(summary_df: pd.DataFrame) -> go.Figure:
    donut_df = summary_df.head(8).copy()
    if len(summary_df) > 8:
        others_total = int(summary_df.iloc[8:]["Total Complaints"].sum())
        if others_total > 0:
            donut_df = pd.concat(
                [donut_df, pd.DataFrame([{"Civic Agency": "Other Departments", "Total Complaints": others_total}])],
                ignore_index=True,
            )
    fig = px.pie(
        donut_df,
        names="Civic Agency",
        values="Total Complaints",
        hole=0.62,
        color_discrete_sequence=[
            "#12335c",
            "#0f766e",
            "#1d4ed8",
            "#f59e0b",
            "#b42318",
            "#7c3aed",
            "#0f172a",
            "#475569",
            "#94a3b8",
        ],
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_severity_donut(df: pd.DataFrame) -> go.Figure:
    severity_df = (
        df["severity"]
        .value_counts()
        .reindex(SEVERITY_ORDER + ["Unknown"], fill_value=0)
        .reset_index()
    )
    severity_df.columns = ["Severity", "Count"]
    severity_df = severity_df[severity_df["Count"] > 0]
    fig = px.pie(
        severity_df,
        names="Severity",
        values="Count",
        hole=0.62,
        color="Severity",
        color_discrete_map=SEVERITY_COLORS,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_last_7_day_trend(recent_df: pd.DataFrame) -> go.Figure:
    trend_df = recent_df.groupby(["complaint_date", "severity"]).size().reset_index(name="count")
    fig = px.line(
        trend_df,
        x="complaint_date",
        y="count",
        color="severity",
        markers=True,
        category_orders={"severity": SEVERITY_ORDER},
        color_discrete_map=SEVERITY_COLORS,
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Complaint Date",
        yaxis_title="Complaint Count",
        legend_title="Severity",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_yearly_trend(df: pd.DataFrame) -> go.Figure:
    yearly_df = df.dropna(subset=["year"]).groupby("year").size().reset_index(name="count")
    fig = px.line(
        yearly_df,
        x="year",
        y="count",
        markers=True,
        color_discrete_sequence=["#12335c"],
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Year",
        yaxis_title="Complaint Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_yearly_severity_breakdown(df: pd.DataFrame) -> go.Figure:
    yearly_severity_df = (
        df.dropna(subset=["year"])
        .groupby(["year", "severity"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        yearly_severity_df,
        x="year",
        y="count",
        color="severity",
        category_orders={"severity": SEVERITY_ORDER},
        color_discrete_map=SEVERITY_COLORS,
    )
    fig.update_layout(
        barmode="stack",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Year",
        yaxis_title="Complaint Count",
        legend_title="Severity",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_style() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f5f7fb;
                --surface: rgba(255, 255, 255, 0.96);
                --surface-strong: #ffffff;
                --ink: #102136;
                --muted: #62738c;
                --line: rgba(16, 33, 54, 0.08);
                --navy: #12335c;
                --teal: #0f766e;
                --soft-shadow: 0 14px 34px rgba(16, 33, 54, 0.08);
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 26%),
                    radial-gradient(circle at top right, rgba(18, 51, 92, 0.08), transparent 24%),
                    linear-gradient(180deg, #fbfcff 0%, var(--bg) 100%);
            }
            .block-container {
                padding-top: 1.15rem;
                padding-bottom: 2rem;
                max-width: 96rem;
            }
            .hero-shell {
                padding: 1.6rem 1.8rem;
                border-radius: 1.35rem;
                background: linear-gradient(135deg, #102136 0%, #173d6d 48%, #0f766e 100%);
                color: white;
                box-shadow: 0 20px 42px rgba(16, 33, 54, 0.18);
                margin-bottom: 1.1rem;
            }
            .hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.82fr);
                gap: 1.2rem;
                align-items: stretch;
            }
            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 0.74rem;
                opacity: 0.82;
                margin-bottom: 0.35rem;
            }
            .hero-shell h1 {
                margin: 0;
                font-size: 2.4rem;
                line-height: 1.05;
            }
            .hero-shell p {
                margin: 0.72rem 0 0 0;
                max-width: 62ch;
                color: rgba(255, 255, 255, 0.88);
                line-height: 1.6;
            }
            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1rem;
            }
            .pill {
                display: inline-flex;
                align-items: center;
                padding: 0.42rem 0.78rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.18);
                font-size: 0.82rem;
                font-weight: 600;
            }
            .hero-panel {
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 1.05rem;
                padding: 1rem;
                backdrop-filter: blur(10px);
            }
            .hero-panel-title {
                font-size: 0.92rem;
                font-weight: 700;
                margin-bottom: 0.72rem;
            }
            .hero-panel ul {
                margin: 0;
                padding-left: 1.1rem;
                line-height: 1.65;
                color: rgba(255, 255, 255, 0.9);
            }
            .sidebar-badge {
                padding: 0.55rem 0.72rem;
                border-radius: 0.75rem;
                font-weight: 700;
                text-align: center;
                margin: 0.55rem 0 0.75rem 0;
            }
            .sidebar-badge.success {
                background: rgba(34, 197, 94, 0.12);
                color: #15803d;
                border: 1px solid rgba(34, 197, 94, 0.22);
            }
            .sidebar-badge.danger {
                background: rgba(239, 68, 68, 0.10);
                color: #b91c1c;
                border: 1px solid rgba(239, 68, 68, 0.20);
            }
            .section-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 1.15rem;
                box-shadow: var(--soft-shadow);
                padding: 1rem 1rem 0.35rem 1rem;
                margin-bottom: 1rem;
            }
            .section-title {
                font-size: 1rem;
                font-weight: 750;
                color: var(--ink);
                margin-bottom: 0.65rem;
            }
            .metric-card {
                background: var(--surface-strong);
                border: 1px solid var(--line);
                border-radius: 1rem;
                box-shadow: var(--soft-shadow);
                padding: 0.95rem 1rem;
                min-height: 158px;
            }
            .metric-card h4 {
                margin: 0;
                font-size: 0.82rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: var(--teal);
            }
            .metric-card h3 {
                margin: 0.5rem 0 0.7rem 0;
                font-size: 1.7rem;
                color: var(--ink);
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.45rem 0.8rem;
                font-size: 0.9rem;
                color: var(--muted);
            }
            .submission-card {
                background: rgba(255, 255, 255, 0.97);
                border: 1px solid var(--line);
                border-radius: 1.15rem;
                box-shadow: var(--soft-shadow);
                padding: 1rem 1rem 0.5rem 1rem;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(backend_url: str) -> str:
    st.sidebar.markdown("## Civic Control Center")
    st.sidebar.caption("Municipal grievance monitoring, routing, and escalation")
    page = st.sidebar.radio(
        "Navigation",
        ["Operations Dashboard", "Register Complaint"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backend Status")
    st.sidebar.write(normalize_backend_url(backend_url))

    launch_status = st.session_state.get("backend_launch_status")
    if launch_status:
        st.sidebar.info(str(launch_status))

    status = st.session_state.get("backend_status")
    if status:
        ok, message = status
        if ok:
            st.sidebar.markdown('<div class="sidebar-badge success">Prediction services online</div>', unsafe_allow_html=True)
            st.sidebar.success(message)
        else:
            st.sidebar.markdown('<div class="sidebar-badge danger">Prediction services offline</div>', unsafe_allow_html=True)
            st.sidebar.error(message)

    if st.sidebar.button("Recheck backend"):
        ok, message = ensure_backend_running(backend_url)
        st.session_state["backend_status"] = (ok, message)
        st.rerun()

    st.sidebar.caption("New complaints registered here update the dashboard counts and charts immediately.")
    return page


def render_hero(df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    total_complaints = len(df)
    critical_count = int((df["severity"] == "Critical").sum())
    top_department = "No department available"
    top_department_count = 0
    if not summary_df.empty:
        top_department = str(summary_df.iloc[0]["Civic Agency"])
        top_department_count = int(summary_df.iloc[0]["Total Complaints"])

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Municipal Grievance Monitoring Platform</div>
                    <h1>Complaint Routing and Escalation Dashboard</h1>
                    <p>
                        Monitor civic complaint volume across departments, track severity pressure,
                        and register new grievances through a production-style control center built
                        for operational teams.
                    </p>
                    <div class="badge-row">
                        <span class="pill">Total complaints: {total_complaints:,}</span>
                        <span class="pill">Critical complaints: {critical_count:,}</span>
                        <span class="pill">Highest load: {top_department} ({top_department_count})</span>
                    </div>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-title">Operational priorities</div>
                    <ul>
                        <li>Classify complaints to the exact civic agency.</li>
                        <li>Rank each grievance by severity for escalation.</li>
                        <li>Refresh department and severity analytics in real time.</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_department_cards(summary_df: pd.DataFrame) -> None:
    st.markdown("### Department-wise Complaint Dashboard")
    if summary_df.empty:
        st.info("No department analytics are available.")
        return

    for start in range(0, min(len(summary_df), 9), 3):
        row_df = summary_df.iloc[start : start + 3]
        cols = st.columns(len(row_df))
        for idx, (_, row) in enumerate(row_df.iterrows()):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>{row["Civic Agency"]}</h4>
                        <h3>{int(row["Total Complaints"]):,}</h3>
                        <div class="metric-grid">
                            <div>Critical</div><div>{int(row["Critical"])}</div>
                            <div>High</div><div>{int(row["High"])}</div>
                            <div>Medium</div><div>{int(row["Medium"])}</div>
                            <div>Low</div><div>{int(row["Low"])}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_kpis(df: pd.DataFrame, summary_df: pd.DataFrame, recent_df: pd.DataFrame) -> None:
    active_complaints = int(df["severity"].isin(["Critical", "High"]).sum())
    critical_count = int((df["severity"] == "Critical").sum())
    departments_covered = int(summary_df["Civic Agency"].nunique()) if not summary_df.empty else 0
    recent_total = len(recent_df)

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
    kpi_1.metric("Total Complaints", f"{len(df):,}")
    kpi_2.metric("Active Complaints", f"{active_complaints:,}")
    kpi_3.metric("Critical Complaints", f"{critical_count:,}")
    kpi_4.metric("Departments Covered", f"{departments_covered}")
    kpi_5.metric("Last 7 Complaint Dates", f"{recent_total:,}")


def render_time_analytics(df: pd.DataFrame) -> None:
    recent_df, recent_dates = build_recent_seven_day_window(df)
    st.markdown("### Time-based Analytics")
    info_col_1, info_col_2 = st.columns([1.4, 1])
    with info_col_1:
        st.info(
            "Last 7 complaint dates are derived from the most recent seven distinct complaint dates available in the historical dataset and newly registered cases."
        )
    with info_col_2:
        st.metric("Complaints in Last 7 Complaint Dates", f"{len(recent_df):,}")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown('<div class="section-card"><div class="section-title">Severity Trend Over Last 7 Complaint Dates</div>', unsafe_allow_html=True)
        if recent_df.empty or not recent_dates:
            st.info("No complaint dates are available for recent trend analysis.")
        else:
            st.plotly_chart(build_last_7_day_trend(recent_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_2:
        st.markdown('<div class="section-card"><div class="section-title">Yearly Complaint Trend</div>', unsafe_allow_html=True)
        if df.dropna(subset=["year"]).empty:
            st.info("No yearly complaint data is available.")
        else:
            st.plotly_chart(build_yearly_trend(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Yearly Severity Breakdown</div>', unsafe_allow_html=True)
    if df.dropna(subset=["year"]).empty:
        st.info("No yearly severity data is available.")
    else:
        st.plotly_chart(build_yearly_severity_breakdown(df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_dashboard(df: pd.DataFrame) -> None:
    summary_df = build_department_summary(df)
    recent_df, _ = build_recent_seven_day_window(df)

    render_hero(df, summary_df)
    render_kpis(df, summary_df, recent_df)
    render_department_cards(summary_df)

    chart_col_1, chart_col_2 = st.columns([1.45, 1])
    with chart_col_1:
        st.markdown('<div class="section-card"><div class="section-title">Severity Distribution Per Department</div>', unsafe_allow_html=True)
        st.plotly_chart(build_department_bar(summary_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_col_2:
        st.markdown('<div class="section-card"><div class="section-title">Overall Complaint Distribution by Department</div>', unsafe_allow_html=True)
        st.plotly_chart(build_department_donut(summary_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    chart_col_3, chart_col_4 = st.columns([1.1, 1.35])
    with chart_col_3:
        st.markdown('<div class="section-card"><div class="section-title">Overall Complaint Distribution by Severity</div>', unsafe_allow_html=True)
        st.plotly_chart(build_severity_donut(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_col_4:
        st.markdown('<div class="section-card"><div class="section-title">Department Summary Table</div>', unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    render_time_analytics(df)

    recent_registrations = df[df["source"] == "registered_via_ui"].sort_values("created_at", ascending=False).head(12)
    st.markdown('<div class="section-card"><div class="section-title">Recently Registered Complaints</div>', unsafe_allow_html=True)
    if recent_registrations.empty:
        st.info("No complaints have been registered through this interface yet.")
    else:
        display = recent_registrations.copy()
        display["description"] = display["description"].apply(lambda value: truncate_text(str(value), 120))
        display["created_at"] = display["created_at"].dt.tz_convert("Asia/Calcutta").dt.strftime("%Y-%m-%d %H:%M")
        display = display.rename(
            columns={
                "case_id": "Case ID",
                "created_at": "Registered At",
                "description": "Complaint",
                "civic_agency_title": "Civic Agency",
                "severity": "Severity",
            }
        )
        st.dataframe(
            display[["Case ID", "Registered At", "Civic Agency", "Severity", "Complaint"]],
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_registration_page(backend_url: str) -> None:
    st.markdown(
        """
        <div class="submission-card">
            <h2 style="margin-top:0; color:#102136;">Register New Civic Complaint</h2>
            <p style="color:#62738c; margin-top:0;">
                Submit a new grievance to classify the responsible civic agency and assign a severity level.
                Successful registrations are appended to the operational dataset and reflected across the dashboard immediately.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("complaint_registration_form"):
        complaint_text = st.text_area(
            "Complaint description",
            placeholder="Describe the grievance in clear operational detail...",
            height=220,
            max_chars=1000,
        )
        submitted = st.form_submit_button("Register Complaint")

    if submitted:
        if not complaint_text.strip():
            st.warning("Please enter a complaint before registering it.")
        else:
            with st.spinner("Routing complaint and ranking severity..."):
                try:
                    prediction = call_prediction_api(backend_url, complaint_text)
                    record = register_new_complaint(complaint_text, prediction)
                except requests.RequestException as exc:
                    st.error(f"Could not reach the backend: {exc}")
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    result_col_1, result_col_2, result_col_3 = st.columns(3)
                    result_col_1.metric("Assigned Civic Agency", str(prediction["predicted_department"]))
                    result_col_2.metric("Assigned Severity", normalize_severity(prediction["severity"]))
                    result_col_3.metric("Case Reference", record["case_id"])
                    st.success("Complaint registered successfully. Dashboard counts and charts were updated.")
                    st.caption(
                        f"Registered at {record['created_at']} | backend response time: {float(prediction['elapsed_seconds']):.2f} seconds"
                    )
                    st.markdown("### Complaint Summary")
                    st.write(complaint_text.strip())

    latest_registration = st.session_state.get("latest_registration")
    if latest_registration:
        st.markdown("### Most Recent Registration")
        recent_col_1, recent_col_2 = st.columns(2)
        with recent_col_1:
            st.info(f"Case ID: {latest_registration['case_id']}")
            st.info(f"Civic Agency: {latest_registration['civic_agency_title']}")
        with recent_col_2:
            st.info(f"Severity: {latest_registration['severity']}")
            st.info(f"Registered At: {latest_registration['created_at']}")


def main() -> None:
    st.set_page_config(page_title="Grievance Redressal System", layout="wide")
    initialize_dashboard_state()
    render_style()

    backend_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_BACKEND_URL)
    backend_url = normalize_backend_url(backend_url) or DEFAULT_BACKEND_URL
    backend_ok, backend_message = ensure_backend_running(backend_url)
    st.session_state["backend_status"] = (backend_ok, backend_message)
    page = render_sidebar(backend_url)

    if page == "Operations Dashboard":
        render_dashboard(st.session_state.dashboard_df)
    else:
        render_registration_page(backend_url)


if __name__ == "__main__":
    main()
