# Asset Health Agentic AI Fault Detection Dashboard

This Streamlit app plots equipment time series at the top of the screen, lists fault times and maintenance schedules on the bottom left, and shows an agentic AI diagnosis and recommended resolution on the bottom right.

## Run

```powershell
streamlit run app.py
```

## Data format

Upload a CSV with a timestamp column plus numeric sensor columns. If no CSV is uploaded, the app uses a built-in demo historian trace with two injected anomaly scenarios.

Expected columns can include:

- timestamp
- motor_current
- vibration
- temperature
- pressure
- flow_rate

The diagnosis layer is rule-based and explainable, so it can be tuned to the exact MeCoE asset classes, fault library, and work-order logic from the proposal.

## Extension points

The current version uses an explainable local agent workflow: monitoring, diagnosis, scheduling, and resolution. It can be extended with an LLM endpoint, CMMS work-order creation, or MeCoE-specific fault libraries once those interfaces are available.
