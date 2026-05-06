from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep
from urllib.parse import urlparse

import pandas as pd
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
DEFAULT_BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")
LOOKUP_MAX_ROW = 17145
API_TIMEOUT_SECONDS = 20
BACKEND_STARTUP_TIMEOUT_SECONDS = 30


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


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "augmented_combined.csv",
        usecols=["description", "civic_agency_title", "severity"],
    )


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


def truncate_for_backend(text: str, limit: int = 1000) -> tuple[str, bool]:
    complaint = str(text)
    if len(complaint) <= limit:
        return complaint, False
    return complaint[:limit], True


def call_prediction_api(base_url: str, complaint: str) -> dict[str, str | float]:
    start = perf_counter()
    payload = {"complaint": complaint}
    response = requests.post(
        build_prediction_url(base_url),
        json=payload,
        timeout=API_TIMEOUT_SECONDS,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = "Prediction request failed."
        try:
            error_body = response.json()
            if isinstance(error_body, dict) and error_body.get("detail"):
                detail = str(error_body["detail"])
        except Exception:
            if response.text:
                detail = response.text.strip()
        raise RuntimeError(detail) from exc

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response from FastAPI backend.")

    department = data.get("predicted_department")
    severity = data.get("severity")
    if not department or not severity:
        raise RuntimeError("FastAPI backend returned an incomplete prediction.")

    return {
        "predicted_department": str(department),
        "severity": str(severity),
        "elapsed_seconds": perf_counter() - start,
    }


def safe_text(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return str(value)


def render_prediction_cards(
    prediction: dict[str, str | float],
    actual_department: str | None = None,
    actual_severity: str | None = None,
) -> None:
    dept_col, sev_col = st.columns(2)

    with dept_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-kicker">Civic Authority</div>
                <div class="result-value">{prediction["predicted_department"]}</div>
                <div class="result-caption">Backend department prediction from FastAPI.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if actual_department is not None:
            st.caption(f"Actual department: {actual_department}")

    with sev_col:
        severity_value = str(prediction["severity"]).title()
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-kicker">Severity</div>
                <div class="result-value">{severity_value}</div>
                <div class="result-caption">Backend severity prediction from FastAPI.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if actual_severity is not None:
            st.caption(f"Actual severity: {actual_severity}")

    with st.expander("Raw backend response"):
        st.json(
            {
                "predicted_department": prediction["predicted_department"],
                "severity": prediction["severity"],
                "elapsed_seconds": round(float(prediction["elapsed_seconds"]), 3),
            }
        )


def render_sidebar(backend_url: str) -> None:
    st.sidebar.subheader("FastAPI Backend")
    st.sidebar.caption("Streamlit uses FastAPI for predictions. Swagger remains for developer testing.")
    st.sidebar.write(normalize_backend_url(backend_url))

    launch_status = st.session_state.get("backend_launch_status")
    if launch_status:
        st.sidebar.info(str(launch_status))

    if st.sidebar.button("Check backend connection"):
        ok, message = ensure_backend_running(backend_url)
        st.session_state["backend_status"] = (ok, message)

    status = st.session_state.get("backend_status")
    if status:
        ok, message = status
        if ok:
            st.sidebar.markdown(
                '<div class="sidebar-badge success">Backend connected</div>',
                unsafe_allow_html=True,
            )
            st.sidebar.success(message)
        else:
            st.sidebar.markdown(
                '<div class="sidebar-badge danger">Backend offline</div>',
                unsafe_allow_html=True,
            )
            st.sidebar.error(message)

    st.sidebar.caption("Prediction endpoint: `POST /predict`")
    st.sidebar.caption("Validation: complaint is required and limited to 1000 characters.")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">Client Front End</div>
                    <h1>AI Grievance System</h1>
                    <p>
                        A polished complaint screening interface for non-technical users.
                        Streamlit handles presentation, while FastAPI provides the predictions.
                    </p>
                    <div class="badge-row">
                        <span class="pill">FastAPI-backed</span>
                        <span class="pill">Department + severity</span>
                        <span class="pill">Dataset lookup ready</span>
                    </div>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-title">What this app does</div>
                    <ul>
                        <li>Accepts a complaint description from the user.</li>
                        <li>Calls the FastAPI backend for predictions.</li>
                        <li>Shows department and severity side by side.</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_strip() -> None:
    cols = st.columns(3)
    cards = [
        ("Client-facing", "Streamlit presents the interface, so end users never touch Swagger."),
        ("Backend powered", "All predictions come from the FastAPI `/predict` endpoint."),
        ("Production-aware", "The UI can point to any backend URL through the sidebar."),
    ]

    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-title">{title}</div>
                    <div class="feature-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="AI Grievance System", layout="wide")

    st.markdown(
        """
        <style>
            :root {
                --bg: #f5f7fb;
                --ink: #0f172a;
                --muted: #64748b;
                --card: rgba(255, 255, 255, 0.92);
                --line: rgba(15, 23, 42, 0.08);
                --accent: #0f766e;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 28%),
                    radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 24%),
                    linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
                color: var(--ink);
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }
            .hero-shell {
                padding: 1.5rem 1.75rem;
                border-radius: 1.4rem;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #0f766e 100%);
                color: white;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
                margin-bottom: 1rem;
            }
            .hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.8fr);
                gap: 1.2rem;
                align-items: stretch;
            }
            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 0.74rem;
                opacity: 0.8;
                margin-bottom: 0.35rem;
            }
            .hero-shell h1 {
                margin: 0;
                font-size: 2.35rem;
                line-height: 1.05;
            }
            .hero-shell p {
                margin: 0.65rem 0 0 0;
                max-width: 58ch;
                color: rgba(255, 255, 255, 0.86);
                font-size: 1rem;
                line-height: 1.55;
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
                padding: 0.42rem 0.72rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.16);
                font-size: 0.82rem;
                font-weight: 600;
            }
            .hero-panel {
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 1.1rem;
                padding: 1rem 1rem 0.9rem 1rem;
                backdrop-filter: blur(10px);
            }
            .hero-panel-title {
                font-size: 0.9rem;
                font-weight: 700;
                margin-bottom: 0.7rem;
            }
            .hero-panel ul {
                margin: 0;
                padding-left: 1.1rem;
                color: rgba(255, 255, 255, 0.88);
                line-height: 1.6;
            }
            .feature-card,
            .result-card {
                background: var(--card);
                border: 1px solid var(--line);
                border-radius: 1rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            }
            .feature-card {
                padding: 1rem 1rem 0.95rem 1rem;
                min-height: 118px;
            }
            .feature-title,
            .result-kicker {
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--accent);
                margin-bottom: 0.45rem;
            }
            .feature-body,
            .result-caption {
                color: var(--muted);
                line-height: 1.55;
            }
            .result-card {
                padding: 1rem;
                margin-bottom: 0.35rem;
            }
            .result-value {
                font-size: 1.55rem;
                font-weight: 800;
                color: var(--ink);
                margin-bottom: 0.3rem;
                word-break: break-word;
            }
            .sidebar-badge {
                padding: 0.55rem 0.7rem;
                border-radius: 0.75rem;
                font-weight: 700;
                text-align: center;
                margin: 0.5rem 0 0.75rem 0;
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
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.4rem;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 0.6rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_hero()
    render_feature_strip()

    backend_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_BACKEND_URL)
    backend_url = normalize_backend_url(backend_url) or DEFAULT_BACKEND_URL
    backend_ok, backend_message = ensure_backend_running(backend_url)
    st.session_state["backend_status"] = (backend_ok, backend_message)
    render_sidebar(backend_url)

    dataset = load_dataset()
    if dataset.empty:
        st.error("The dataset is empty, so dataset lookup cannot run.")
        return

    max_row = min(LOOKUP_MAX_ROW, len(dataset) - 1)
    new_tab, dataset_tab = st.tabs(["New Complaint Testing", "Dataset Complaint Lookup"])

    with new_tab:
        st.subheader("New Complaint Testing")
        st.write("Type a complaint and let the backend return the predicted department and severity.")

        with st.form("new_complaint_form"):
            complaint_text = st.text_area(
                "Complaint text",
                placeholder="Type the grievance or complaint here...",
                height=220,
                max_chars=1000,
            )
            submitted = st.form_submit_button("Predict from FastAPI")

        if submitted:
            if not complaint_text.strip():
                st.warning("Please enter a complaint before predicting.")
            else:
                with st.spinner("Calling the FastAPI backend..."):
                    try:
                        prediction = call_prediction_api(backend_url, complaint_text)
                    except requests.RequestException as exc:
                        st.error(f"Could not reach the backend: {exc}")
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        st.success("Prediction retrieved from FastAPI.")
                        st.caption(f"Backend round-trip time: {prediction['elapsed_seconds']:.2f} seconds")
                        render_prediction_cards(prediction)

    with dataset_tab:
        st.subheader("Dataset Complaint Lookup")
        st.write("Use a dataset row number to inspect the complaint, actual labels, and backend predictions.")
        st.caption(f"Valid row numbers: 0 to {max_row}")

        with st.form("dataset_lookup_form"):
            row_number = st.number_input(
                "Row number",
                min_value=0,
                max_value=max_row,
                value=0,
                step=1,
            )
            lookup = st.form_submit_button("Lookup")

        if lookup:
            row = dataset.iloc[int(row_number)]
            complaint_text = safe_text(row["description"])
            actual_department = safe_text(row["civic_agency_title"])
            actual_severity = safe_text(row["severity"])

            st.markdown("### Complaint Text")
            st.text_area("description", value=complaint_text, height=240, disabled=True)

            prediction_input, was_truncated = truncate_for_backend(complaint_text)
            if was_truncated:
                st.warning(
                    "This complaint was longer than 1000 characters, so the backend input was truncated to meet the API limit."
                )

            with st.spinner("Calling the FastAPI backend..."):
                try:
                    prediction = call_prediction_api(backend_url, prediction_input)
                except requests.RequestException as exc:
                    st.error(f"Could not reach the backend: {exc}")
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.success("Prediction retrieved from FastAPI.")
                    st.caption(f"Backend round-trip time: {prediction['elapsed_seconds']:.2f} seconds")
                    st.markdown("### Actual Labels")
                    actual_col_1, actual_col_2 = st.columns(2)
                    with actual_col_1:
                        st.info(f"Actual department: {actual_department}")
                    with actual_col_2:
                        st.info(f"Actual severity: {actual_severity}")

                    render_prediction_cards(
                        prediction,
                        actual_department=actual_department,
                        actual_severity=actual_severity,
                    )


if __name__ == "__main__":
    main()
