# Asset Health Agentic AI Fault Detection, Diagnosis and Resolution
This repository contains a production-ready Streamlit dashboard designed for real-time asset reliability monitoring and automated root-cause analysis. The platform leverages robust time-series statistics and a multi-agent orchestration framework to detect process anomalies, diagnose physical fault states, and generate actionable work-order response schedules.

## Technical Architecture
The application implements a decoupled, three-stage pipeline that processes raw historian data into prioritized maintenance actions:
```
[ Raw Historian Data (CSV) ] 
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 1. Statistical Signal Processing Engine                │
│    • Rolling Median Baseline (Window: 36, Min: 12)     │
│    • Median Absolute Deviation (MAD) Calculation       │
│    • Dynamic Robust Z-Score Vectorization              │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Multi-Agent Orchestration Domain                    │
│    • Monitoring Agent : Time-window grouping & trips   │
│    • Diagnosis Agent  : Deterministic pattern matching │
│    • Resolution Agent : SLAs, checklists & actions     │
└────────────────────────────────────────────────────────┘
            │
            ▼
[ Interactive UI Layout (Plotly + Streamlit Metrics) ]
```

## Core Architecture Components
### Statistical Signal Processing Engine:
Instead of using standard standard deviation (which is heavily biased by the anomalies it tries to detect), this system implements a Robust Z-Score calculation derived from a rolling Median Absolute Deviation (MAD). This guarantees high resilience against extreme outliers and signal noise.

### Multi-Agent Decoupled Domain:

**Monitoring Agent: Evaluates vector-wide excursions, tracks active trip counters, and groups continuous anomalous windows together using index proximity thresholds.

**Diagnosis Agent: Evaluates the cross-sectional intersection of highly elevated Z-scores against deterministic engineering failure modes (e.g., Cavitation, Bearing Wear).

**Resolution Agent: Computes asset risk profiles, maps explicit operational response SLAs based on calculated peak structural severity, and populates field checklists.

## Core Processing Workflow
```
       [ Input Data Stream ]
                 │
                 ▼
     [ Compute Rolling Median ]
                 │
                 ▼
       [ Calculate Residuals ]
                 │
                 ▼
       [ Derive Rolling MAD ]
                 │
                 ▼
    [ Apply Epsilon Floor (1e-6) ] ◄── Prevents Zero-Division on Flatlines
                 │
                 ▼
   [ Generate Robust Z-Scores ]
                 │
                 ▼
 ┌───────────────┴───────────────┐
 │ Is Max(|Z_col|) >= Threshold? │
 └───────────────┬───────────────┘
                 │
        ┌────────┴────────┐
       YES                NO
        │                 │
        ▼                 ▼
[ Flag Anomaly ]   [ Normal State ]
        │
        ▼
[ Window Grouping (Gap <= 2) ]
        │
        ▼
[ Multi-Agent Diagnosis & SLA ]
```

Additional Processing Details

**Windowing & Residual Extraction: The system scans each selected signal vector, establishing a central baseline using a centered rolling median window of 36 periods. The raw signal is subtracted from this baseline to yield a clean residual array.

**Dynamic Scaling (MAD): The rolling median of absolute residuals is computed and scaled by the constant factor (1.4826) to align asymptotically with standard normal distributions.

**Flatline Protection: If a sensor flatlines or stops reporting data variation, the denominator (scaled_mad) automatically defaults to a minute epsilon floor (ϵ=1×10 
−6 ). This guards against ZeroDivisionError tracking while preserving sensitivity to sudden data changes immediately following a dead-band zone.

**Temporal Clumping: Timestamps flagged as anomalous are assembled into discrete events. If two anomalous timestamps occur within 2 samples of each other, they are merged into a single continuous plant event window.

**Heuristic Fault Intersection: The system scans the average Z-scores within the active event window. Signals demonstrating an average score above 55% of the master trip threshold are marked as "Active Trigger Signals." These triggers are passed directly to the rule engine to determine the closest matching fault profile.

## Monitored Fault Rules & Engineering Conditions
The diagnostic agent evaluates five core telemetry streams (motor_current, vibration, temperature, pressure, and flow_rate) across four hardcoded fault signatures:

| Fault Mode | Signature Criteria | Action Checklist Includes |
| --- | --- | --- |
| Bearing wear / lubrication loss | High vibration + rising temperature + elevated current | Vibration spectrum analysis, lubrication history check |
| Pump cavitation / suction restriction | Pressure instability + reduced flow + elevated vibration | Strainer inspection, NPSH margin verification |
| Cooling degradation / thermal overload | Temperature excursion + sustained high current | Fan and heat exchanger evaluation, current balancing |
| Instrumentation drift / fault | Cross-signal disagreement without matching process loop response | Calibration validation, redundant tag verification |

## Getting Started
Prerequisites
Ensure you have a python environment configured with the required dependencies:

```
pip install streamlit pandas numpy plotly pyarrow
```

Running the Platform
To launch the multi-agent execution dashboard locally, execute:

```
streamlit run app.py
```

## Using the App
Load Data: The system spins up with pre-packaged synthetic demonstration data demonstrating concurrent bearing failure and cavitation sequences. You can upload any standard plant historian CSV with a timestamp column.

Tune Sensitivity: Use the sidebar slider to manipulate the robust trip threshold. Lowering this value captures subtle, early-stage signal degradation; increasing it ensures warnings are restricted to critical process excursions.

Execute Actions: Track automated work-order schedules on the bottom-left panel, and complete agent-generated physical checks on the bottom-right panel to clear active anomalies.

# MIT License
Copyright (c) 2026 CK

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
