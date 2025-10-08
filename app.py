# Falcon Eye 360 — Offline Video MVP (Streamlit UI v0)
# Single-file Streamlit skeleton: Upload/Select → Review Run → Settings
# This version is code-only UI scaffolding with mock data.
# Later you will plug in the trained models and the real pipeline.

import io
import json
import time
import uuid
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Falcon Eye 360 (Offline)", layout="wide")

DEFAULTS = {
    "accident_score_thresh": 0.35,
    "merge_gap_frames": 12,
    "min_iou_nearmiss": 0.20,
    "min_delta_speed": 2.5,  # m/s proxy
    "blur_strength": 21,
    "blur_margin_px": 8,
    "forecast_horizon_min": 10,
    "bin_size_sec": 10,
    "device": "cpu",  # or "cuda:0"
}

# Keep simple state
if "run" not in st.session_state:
    st.session_state.run = None

# ---------------------------
# Helper: create stub outputs
# ---------------------------

def _stub_run_artifacts(video_path: Path) -> dict:
    """Simulate a completed analysis and return artifacts in-memory.
    In v1 you will replace this with the real pipeline call.
    """
    run_id = str(uuid.uuid4())[:8]
    # Fake events
    events = [
        {
            "event_id": 1,
            "type": "accident",
            "start_time": 3.4,
            "severity": "Moderate",
            "confidence": 0.78,
            "involved_track_ids": [5, 12],
            "peak_frame": 87,
        },
        {
            "event_id": 2,
            "type": "near_miss",
            "start_time": 11.2,
            "severity": "Minor",
            "confidence": 0.66,
            "involved_track_ids": [2, 9],
            "peak_frame": 286,
        },
    ]

    events_df = pd.DataFrame([
        {
            "Start (s)": e["start_time"],
            "Type": e["type"],
            "Severity": e["severity"],
            "Confidence": round(float(e["confidence"]), 2),
            "Involved IDs": ",".join(map(str, e["involved_track_ids"]))
        }
        for e in events
    ])

    # Fake time series & forecast
    t = np.arange(0, 120 + 1, DEFAULTS["bin_size_sec"])  # 2 minutes at 10s bins
    counts = (2 + 0.03 * t + np.random.randn(t.size) * 0.2).clip(min=0)
    forecast_steps = int(DEFAULTS["forecast_horizon_min"] * 60 / DEFAULTS["bin_size_sec"])
    fut_t = np.arange(t[-1] + DEFAULTS["bin_size_sec"],
                      t[-1] + (forecast_steps + 1) * DEFAULTS["bin_size_sec"],
                      DEFAULTS["bin_size_sec"])
    # naïve persistence + small drift
    forecast = np.full(fut_t.shape, counts[-1]) + 0.01 * np.arange(forecast_steps)

    ts_df = pd.DataFrame({"t_sec": t, "count": counts})
    fc_df = pd.DataFrame({"t_sec": fut_t, "forecast": forecast})

    # JSON artifacts
    events_json = {
        "video": str(video_path.name),
        "events": events,
        "settings": DEFAULTS,
    }
    run_report = {
        "run_id": run_id,
        "frames": 3000,
        "processing_fps": 18.4,
        "num_events": len(events),
        "device": DEFAULTS["device"],
    }

    return {
        "run_id": run_id,
        "events_df": events_df,
        "events_json": json.dumps(events_json, indent=2).encode(),
        "ts_df": ts_df,
        "fc_df": fc_df,
        "run_report": json.dumps(run_report, indent=2).encode(),
    }

# ---------------------------
# Sidebar — Upload/Select & Settings
# ---------------------------
with st.sidebar:
    st.header("Input & Controls")
    tab_in, tab_cfg = st.tabs(["Upload / Select", "Settings"])

    with tab_in:
        uploaded = st.file_uploader("Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
        text_path = st.text_input("Or provide a local video path", placeholder="/path/to/video.mp4")
        export_video = st.checkbox("Export annotated video", True)
        export_logs = st.checkbox("Save JSON/CSV logs", True)
        run_btn = st.button("Run Analysis", use_container_width=True)

    with tab_cfg:
        st.caption("Analysis thresholds & runtime options")
        DEFAULTS["accident_score_thresh"] = st.slider("Accident score threshold", 0.1, 0.9, DEFAULTS["accident_score_thresh"], 0.01)
        DEFAULTS["merge_gap_frames"] = st.slider("Event merge gap (frames)", 2, 60, DEFAULTS["merge_gap_frames"], 1)
        DEFAULTS["min_iou_nearmiss"] = st.slider("Near-miss min IoU", 0.0, 0.8, DEFAULTS["min_iou_nearmiss"], 0.01)
        DEFAULTS["min_delta_speed"] = st.slider("Δspeed threshold (m/s)", 0.5, 8.0, DEFAULTS["min_delta_speed"], 0.1)
        DEFAULTS["blur_strength"] = st.slider("Blur strength (kernel)", 5, 51, DEFAULTS["blur_strength"], 2)
        DEFAULTS["blur_margin_px"] = st.slider("Blur margin (px)", 0, 32, DEFAULTS["blur_margin_px"], 1)
        DEFAULTS["forecast_horizon_min"] = st.select_slider("Forecast horizon (min)", options=[5, 10], value=DEFAULTS["forecast_horizon_min"])
        DEFAULTS["bin_size_sec"] = st.select_slider("Bin size (sec)", options=[5, 10, 15], value=DEFAULTS["bin_size_sec"])
        DEFAULTS["device"] = st.selectbox("Device", ["cpu", "cuda:0"], index=0)

# ---------------------------
# Main — Pages: Upload/Review/Downloads
# ---------------------------
st.title("Falcon Eye 360 — Offline Video")

colA, colB = st.columns([2.2, 1.0], gap="large")

with colA:
    st.subheader("Video Preview")
    video_src_path: Path | None = None

    if uploaded is not None:
        # Save to a temp file so Streamlit can play it
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        video_src_path = Path(tmp.name)
        st.video(str(video_src_path))
    elif text_path.strip():
        p = Path(text_path.strip())
        if p.exists() and p.is_file():
            video_src_path = p
            st.video(str(video_src_path))
        else:
            st.info("Provide a valid local path or upload a file to preview.")
    else:
        st.info("Upload a video or enter a local path to begin.")

    if run_btn:
        if video_src_path is None:
            st.warning("Please upload a video or provide a valid path before running analysis.")
        else:
            with st.status("Running offline analysis…", expanded=True) as status:
                st.write("Loading models & configuration…")
                time.sleep(0.5)
                st.write("Parsing video & preparing frames…")
                time.sleep(0.5)
                st.write("Detecting, tracking, forming events, and forecasting…")
                time.sleep(1.0)
                artifacts = _stub_run_artifacts(video_src_path)
                st.session_state.run = {
                    "video": str(video_src_path),
                    "artifacts": artifacts,
                    "export_video": export_video,
                    "export_logs": export_logs,
                }
                status.update(label="Analysis complete", state="complete", expanded=False)

with colB:
    st.subheader("Run Summary")
    if st.session_state.run is None:
        st.info("No run yet. Configure inputs on the left and click **Run Analysis**.")
    else:
        arts = st.session_state.run["artifacts"]
        rpt = json.loads(arts["run_report"].decode())
        st.metric("Run ID", rpt.get("run_id", "-"))
        st.metric("Events", rpt.get("num_events", 0))
        st.metric("Proc. FPS", rpt.get("processing_fps", 0))
        st.metric("Device", rpt.get("device", "cpu"))

        st.divider()
        st.caption("Alerts")
        st.dataframe(arts["events_df"], use_container_width=True, hide_index=True)

        st.divider()
        st.caption("Forecast")
        # Simple two-line chart using built-in line_chart for v0
        hist = arts["ts_df"].set_index("t_sec")
        fut = arts["fc_df"].set_index("t_sec")
        merged = hist.join(fut, how="outer")
        st.line_chart(merged, use_container_width=True)

# ---------------------------
# Downloads
# ---------------------------
st.subheader("Downloads")
if st.session_state.run is None:
    st.info("Outputs will appear here after a run.")
else:
    arts = st.session_state.run["artifacts"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Events JSON", data=arts["events_json"], file_name="events.json", mime="application/json")
    with c2:
        # join history and forecast for convenience
        merged_csv = pd.concat([
            arts["ts_df"].assign(kind="history"),
            arts["fc_df"].assign(kind="forecast")
        ], ignore_index=True).to_csv(index=False).encode()
        st.download_button("Forecast CSV", data=merged_csv, file_name="forecast.csv", mime="text/csv")
    with c3:
        st.download_button("Run Report", data=arts["run_report"], file_name="run_report.json", mime="application/json")

st.caption("v0 UI scaffold — plug in your trained models and pipeline where noted.")




