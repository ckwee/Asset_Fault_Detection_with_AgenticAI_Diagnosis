from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AssetHealth Agentic AI Fault Detection & Diagnosis", page_icon="FD", layout="wide")

SENSOR_COLUMNS = ["motor_current", "vibration", "temperature", "pressure", "flow_rate"]

FAULT_RULES = {
    "Bearing wear / lubrication loss": {
        "signals": ["vibration", "temperature", "motor_current"],
        "condition": "High vibration with rising temperature and load current.",
        "actions": [
            "Inspect bearing temperature trend and lubrication record.",
            "Check vibration spectrum for bearing defect frequencies.",
            "Schedule lubrication or bearing inspection at the next safe window.",
        ],
    },
    "Pump cavitation or suction restriction": {
        "signals": ["pressure", "flow_rate", "vibration"],
        "condition": "Pressure instability, reduced flow, and elevated vibration.",
        "actions": [
            "Check suction strainer, valve position, and NPSH margin.",
            "Compare inlet pressure against operating envelope.",
            "Reduce load or correct suction-side restriction before escalation.",
        ],
    },
    "Cooling degradation / thermal overload": {
        "signals": ["temperature", "motor_current"],
        "condition": "Temperature excursion with sustained high current.",
        "actions": [
            "Inspect cooling fan, heat exchanger, and ambient condition.",
            "Check load change, duty cycle, and motor current balance.",
            "Plan controlled shutdown if temperature keeps rising.",
        ],
    },
    "Instrumentation drift or transmitter fault": {
        "signals": ["pressure", "flow_rate"],
        "condition": "Signal disagreement without matching process response.",
        "actions": [
            "Validate transmitter calibration and impulse lines.",
            "Cross-check redundant or downstream measurements.",
            "Tag the instrument for calibration if discrepancy persists.",
        ],
    },
}

@dataclass(frozen=True)
class FaultEvent:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    likely_fault: str
    severity: str
    confidence: float
    affected_signals: tuple[str, ...]
    schedule: str
    recommendation: str


def create_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    periods = 288
    ts = pd.date_range("2026-07-02 00:00", periods=periods, freq="5min")
    data = pd.DataFrame({
        "timestamp": ts,
        "motor_current": 71 + rng.normal(0, 1.4, periods),
        "vibration": 2.2 + rng.normal(0, 0.18, periods),
        "temperature": 63 + rng.normal(0, 1.1, periods),
        "pressure": 410 + rng.normal(0, 5.0, periods),
        "flow_rate": 118 + rng.normal(0, 2.4, periods),
    })

    bearing_window = (data.index >= 88) & (data.index <= 116)
    data.loc[bearing_window, "vibration"] += np.linspace(0.9, 3.7, bearing_window.sum())
    data.loc[bearing_window, "temperature"] += np.linspace(2.0, 12.0, bearing_window.sum())
    data.loc[bearing_window, "motor_current"] += np.linspace(1.5, 8.0, bearing_window.sum())

    cavitation_window = (data.index >= 190) & (data.index <= 216)
    data.loc[cavitation_window, "pressure"] -= np.linspace(15.0, 65.0, cavitation_window.sum())
    data.loc[cavitation_window, "flow_rate"] -= np.linspace(5.0, 28.0, cavitation_window.sum())
    data.loc[cavitation_window, "vibration"] += np.linspace(0.6, 2.8, cavitation_window.sum())
    return data


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return create_demo_data()
    data = pd.read_csv(uploaded_file)
    data = data.rename(columns={col: col.strip().lower() for col in data.columns})
    if "timestamp" not in data.columns:
        data = data.rename(columns={data.columns[0]: "timestamp"})
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in [col for col in data.columns if col != "timestamp"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(axis=1, how="all").reset_index(drop=True)


def robust_zscore(series: pd.Series) -> pd.Series:
    rolling_median = series.rolling(36, min_periods=12, center=True).median()
    residual = series - rolling_median
    mad = residual.abs().rolling(36, min_periods=12, center=True).median()
    scaled_mad = (1.4826 * mad).replace(0, np.nan)
    return (residual / scaled_mad).replace([np.inf, -np.inf], np.nan).fillna(0)


def detect_anomalies(data: pd.DataFrame, selected_signals: Iterable[str], threshold: float) -> pd.DataFrame:
    scored = data.copy()
    z_columns = []
    for signal in selected_signals:
        z_col = f"{signal}_z"
        scored[z_col] = robust_zscore(scored[signal])
        z_columns.append(z_col)
    scored["anomaly_score"] = scored[z_columns].abs().max(axis=1) if z_columns else 0
    scored["is_anomaly"] = scored["anomaly_score"] >= threshold
    return scored


def infer_fault(active_signals: tuple[str, ...]) -> str:
    active = set(active_signals)
    best_fault = "Unclassified process anomaly"
    best_overlap = 0
    for fault, detail in FAULT_RULES.items():
        overlap = len(active.intersection(detail["signals"]))
        if overlap > best_overlap:
            best_fault = fault
            best_overlap = overlap
    return best_fault


def schedule_response(severity: str, start_time: pd.Timestamp) -> str:
    if severity == "Critical":
        due = start_time + timedelta(hours=2)
        return f"Immediate response, target before {due:%Y-%m-%d %H:%M}"
    if severity == "High":
        due = start_time + timedelta(hours=12)
        return f"Maintenance window within 12 hours, target {due:%Y-%m-%d %H:%M}"
    due = start_time + timedelta(days=1)
    return f"Monitor and inspect within 24 hours, target {due:%Y-%m-%d %H:%M}"


def build_recommendation(fault: str, severity: str) -> str:
    if fault not in FAULT_RULES:
        return "Dispatch reliability review: compare historian tags, recent work orders, and operator logs before assigning corrective work."
    lead = "Escalate now" if severity == "Critical" else "Recommended next step"
    return f"{lead}: {FAULT_RULES[fault]['actions'][0]}"


def group_anomalies(scored: pd.DataFrame, selected_signals: list[str], threshold: float) -> list[FaultEvent]:
    events: list[FaultEvent] = []
    anomaly_index = scored.index[scored["is_anomaly"]].tolist()
    if not anomaly_index:
        return events

    groups = [[anomaly_index[0]]]
    for idx in anomaly_index[1:]:
        if idx - groups[-1][-1] <= 2:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    for group in groups:
        window = scored.loc[group]
        z_means = {signal: float(window[f"{signal}_z"].abs().mean()) for signal in selected_signals if f"{signal}_z" in window}
        active_signals = tuple(signal for signal, score in sorted(z_means.items(), key=lambda item: item[1], reverse=True) if score >= threshold * 0.55)
        likely_fault = infer_fault(active_signals)
        max_score = float(window["anomaly_score"].max())
        severity = "Critical" if max_score >= threshold + 2 else "High" if max_score >= threshold + 1 else "Moderate"
        confidence = min(0.96, 0.55 + (max_score / 12) + (0.05 * len(active_signals)))
        start_time = pd.Timestamp(window["timestamp"].iloc[0])
        end_time = pd.Timestamp(window["timestamp"].iloc[-1])
        events.append(FaultEvent(start_time, end_time, likely_fault, severity, confidence, active_signals, schedule_response(severity, start_time), build_recommendation(likely_fault, severity)))
    return events


def make_timeseries_chart(scored: pd.DataFrame, signals: list[str], events: list[FaultEvent]) -> go.Figure:
    fig = go.Figure()
    for signal in signals:
        fig.add_trace(go.Scatter(x=scored["timestamp"], y=scored[signal], mode="lines", name=signal.replace("_", " ").title(), line={"width": 2}))
    anomaly_points = scored[scored["is_anomaly"]]
    if not anomaly_points.empty:
        fig.add_trace(go.Scatter(x=anomaly_points["timestamp"], y=anomaly_points[signals[0]], mode="markers", name="Anomaly", marker={"size": 9, "color": "#d92d20", "symbol": "x"}))
    for event in events:
        color = "#d92d20" if event.severity == "Critical" else "#f79009" if event.severity == "High" else "#667085"
        fig.add_vrect(x0=event.start_time, x1=event.end_time, fillcolor=color, opacity=0.18, line_width=0)
    fig.update_layout(height=430, margin={"l": 20, "r": 24, "t": 30, "b": 20}, legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0}, hovermode="x unified", template="plotly_white", xaxis_title="Time", yaxis_title="Sensor value")
    return fig


def event_table(events: list[FaultEvent]) -> pd.DataFrame:
    return pd.DataFrame([{"Fault time": f"{event.start_time:%Y-%m-%d %H:%M}", "Duration": str(event.end_time - event.start_time), "Severity": event.severity, "Likely fault": event.likely_fault, "Schedule": event.schedule} for event in events])


def render_diagnosis(events: list[FaultEvent]) -> None:
    if not events:
        st.info("No active anomaly detected for the current threshold and signal selection.")
        return
    for event in events:
        with st.container(border=True):
            st.subheader(event.likely_fault)
            st.caption(f"{event.severity} | confidence {event.confidence:.0%} | {event.start_time:%Y-%m-%d %H:%M} to {event.end_time:%H:%M}")
            signal_text = ", ".join(signal.replace("_", " ") for signal in event.affected_signals) or "mixed signals"
            st.write(f"**Monitoring agent:** anomaly pattern detected across {signal_text}.")
            st.write(f"**Diagnosis agent:** {FAULT_RULES.get(event.likely_fault, {}).get('condition', 'Agent review required for this mixed anomaly pattern.')}")
            st.write(f"**Resolution agent:** {event.recommendation}")
            if event.likely_fault in FAULT_RULES:
                st.write("**Work order checklist**")
                for action in FAULT_RULES[event.likely_fault]["actions"]:
                    st.checkbox(action, key=f"{event.start_time}-{action}")


def main() -> None:
    st.title("MeCoE Agentic AI Fault Detection, Diagnosis and Resolution")
    with st.sidebar:
        st.header("Data")
        uploaded_file = st.file_uploader("Upload historian CSV", type=["csv"])
        data = load_data(uploaded_file)
        numeric_columns = [col for col in data.columns if col != "timestamp" and pd.api.types.is_numeric_dtype(data[col])]
        default_signals = [col for col in SENSOR_COLUMNS if col in numeric_columns] or numeric_columns[: min(5, len(numeric_columns))]
        selected_signals = st.multiselect("Signals to monitor", numeric_columns, default=default_signals)
        threshold = st.slider("Anomaly sensitivity", min_value=2.0, max_value=6.0, value=3.2, step=0.1)
        st.caption("Lower sensitivity values find more events; higher values only flag stronger excursions.")
    if not selected_signals:
        st.warning("Choose at least one signal to monitor.")
        return
    scored = detect_anomalies(data, selected_signals, threshold)
    events = group_anomalies(scored, selected_signals, threshold)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Signals monitored", len(selected_signals))
    metric_cols[1].metric("Fault events", len(events))
    metric_cols[2].metric("Peak anomaly score", f"{scored['anomaly_score'].max():.1f}")
    metric_cols[3].metric("Latest sample", f"{scored['timestamp'].max():%Y-%m-%d %H:%M}")
    st.plotly_chart(make_timeseries_chart(scored, selected_signals, events), use_container_width=True)
    bottom_left, bottom_right = st.columns([0.92, 1.08], gap="large")
    with bottom_left:
        st.subheader("Fault Time and Schedule")
        table = event_table(events)
        if table.empty:
            st.write("No scheduled fault response for the current view.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
    with bottom_right:
        st.subheader("Agentic AI Diagnosis")
        render_diagnosis(events)

if __name__ == "__main__":
    main()
