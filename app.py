import json
import time
import uuid
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import cv2
from ultralytics import YOLO

# ---------------------------
# Class config (ALL classes)
# ---------------------------
CLASS_NAMES = {
    0: "bike",
    1: "bike_bike_accident",
    2: "bike_object_accident",
    3: "bike_person_accident",
    4: "car",
    5: "car_bike_accident",
    6: "car_car_accident",
    7: "car_object_accident",
    8: "car_person_accident",
    9: "person",
}
VEHICLE_CLASS_IDS   = [0, 4]                           # bike, car
ACCIDENT_CLASS_IDS  = [1, 2, 3, 5, 6, 7, 8]            # all accident subclasses
ALLOWED_CLASSES_SET = set(CLASS_NAMES.keys())          # allow all classes in inference

# ---------------------------
# Paths (defaults to same dir)
# ---------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = APP_DIR / "Accident.pt"
DEFAULT_VIDEO_PATH   = APP_DIR / "AbuDhabiTraffic2.mp4"

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Falcon Eye 360 (Offline)", layout="wide")

DEFAULTS = {
    "accident_score_thresh": 0.35,
    "merge_gap_frames": 12,
    "forecast_horizon_min": 10,
    "bin_size_sec": 10,
    "device": "cpu",      # or "cuda:0"
    "frame_skip": 0,      # process every (frame_skip+1)th frame; 0 = no skip
    "imgsz": 640,         # YOLO inference size
    # ---- Severity rule thresholds (tunable in Settings) ----
    "sev_severe_min_dur_s": 2.5,
    "sev_severe_min_area": 0.10,
    "sev_severe_min_conf": 0.70,
    "sev_moderate_min_dur_s": 1.0,
    "sev_moderate_min_area": 0.05,
    "sev_moderate_min_conf": 0.50,
}

if "run" not in st.session_state:
    st.session_state.run = None

# ---------------------------
# Helpers
# ---------------------------
_YOLO_ACCIDENT_MODEL = None

def resolve_path(p: str | Path) -> Path:
    """Resolve relative paths against the app directory."""
    p = Path(p)
    return p if p.is_absolute() else (APP_DIR / p)

def load_accident_model(weights_path: Path):
    """Load and cache YOLO once per weight file."""
    global _YOLO_ACCIDENT_MODEL
    if _YOLO_ACCIDENT_MODEL is None or getattr(_YOLO_ACCIDENT_MODEL, "_weights_path", "") != str(weights_path):
        m = YOLO(str(weights_path))
        m._weights_path = str(weights_path)
        _YOLO_ACCIDENT_MODEL = m
    return _YOLO_ACCIDENT_MODEL

def _classify_severity(duration_s: float, peak_conf: float, avg_area_ratio: float, d=DEFAULTS) -> str:
    """
    Explainable rule-set:
      - Severe: long duration, high confidence, large area
      - Moderate: medium duration/conf/area
      - Minor: otherwise
    """
    if (duration_s >= d["sev_severe_min_dur_s"]
        and peak_conf >= d["sev_severe_min_conf"]
        and avg_area_ratio >= d["sev_severe_min_area"]):
        return "Severe"
    if (duration_s >= d["sev_moderate_min_dur_s"]
        and peak_conf >= d["sev_moderate_min_conf"]
        and avg_area_ratio >= d["sev_moderate_min_area"]):
        return "Moderate"
    return "Minor"

def _run_yolo_accident(
    video_path: Path,
    weights_path: Path,
    conf_thresh: float,
    device: str,
    merge_gap_frames: int,
    bin_size_sec: int,
    horizon_min: int,
    frame_skip: int,
    imgsz: int,
    export_video: bool,
) -> dict:
    """
    Run detector. Any detection with cls in ACCIDENT_CLASS_IDS is counted as an accident hit/event.
    If export_video=True, writes an annotated MP4 and returns it for download.
    """
    model = load_accident_model(weights_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_area = max(1, width * height)

    # Optional annotated video writer
    writer = None
    tmp_video_path = None
    if export_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_video_path = Path(tmp_video.name)
        tmp_video.close()
        writer = cv2.VideoWriter(str(tmp_video_path), fourcc, fps if fps > 1 else 25.0, (width, height))

    # Progress UI
    progress = st.progress(0, text="Analyzing frames… 0%")
    live_text = st.empty()

    open_event = None
    events = []
    per_second_hits = {}

    processed_frames = 0
    t0 = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_skip > 0 and (frame_idx % (frame_skip + 1)) != 0:
            frame_idx += 1
            continue

        # Run YOLO with class filter (ALL classes)
        res = model.predict(
            frame,
            conf=conf_thresh,
            imgsz=int(imgsz),
            verbose=False,
            device=device,
            classes=list(ALLOWED_CLASSES_SET),
        )

        had_accident = False
        top_conf = 0.0
        top_label = None
        annotated = None
        acc_area_this_frame = 0.0

        if res and len(res):
            r = res[0]
            if export_video:
                annotated = r.plot()  # BGR with drawings
            if r.boxes is not None:
                for b in r.boxes:
                    c = int(b.cls.item()) if b.cls is not None else -1
                    if c not in ALLOWED_CLASSES_SET:
                        continue
                    score = float(b.conf.item() if b.conf is not None else 0.0)
                    # only accident classes influence event + severity
                    if c in ACCIDENT_CLASS_IDS and score >= conf_thresh:
                        had_accident = True
                        if score > top_conf:
                            top_conf = score
                            top_label = CLASS_NAMES.get(c, str(c))
                        # accumulate area ratio for severity
                        if b.xyxy is not None:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                            acc_area_this_frame += (w * h) / frame_area

        # Write annotated frame (or raw if none) when exporting
        if export_video:
            frame_to_write = annotated if annotated is not None else frame
            if writer is not None and frame_to_write is not None and frame_to_write.shape[1] == width and frame_to_write.shape[0] == height:
                writer.write(frame_to_write)

        # Tiny proxy for forecast: count hits per second
        t_sec = int(frame_idx / fps)
        if had_accident:
            per_second_hits[t_sec] = per_second_hits.get(t_sec, 0) + 1

        # Event merge logic + severity accumulators
        if had_accident:
            if open_event is None:
                open_event = {
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "peak_frame": frame_idx,
                    "peak_conf": top_conf,
                    "type": top_label or "accident",
                    # accumulators for severity rules
                    "hit_frames": 1,
                    "area_sum": float(acc_area_this_frame),
                }
            else:
                open_event["end_frame"] = frame_idx
                open_event["hit_frames"] += 1
                open_event["area_sum"] += float(acc_area_this_frame)
                if top_conf > open_event["peak_conf"]:
                    open_event["peak_conf"] = top_conf
                    open_event["peak_frame"] = frame_idx
                    if top_label:
                        open_event["type"] = top_label
        else:
            # close an event when gap exceeds merge_gap_frames
            if open_event is not None and (frame_idx - open_event["end_frame"] > merge_gap_frames):
                events.append(open_event)
                open_event = None

        processed_frames += 1
        frame_idx += 1

        # Progress UI
        if total_frames > 0 and processed_frames % 10 == 0:
            pct = min(1.0, frame_idx / max(1, total_frames))
            progress.progress(int(pct * 100), text=f"Analyzing frames… {int(pct*100)}%")
            live_text.markdown(
                f"**Frame:** {frame_idx}/{total_frames} · **Accident hit:** {'✅' if had_accident else '—'}"
                f"{' (conf=' + str(round(top_conf,3)) + ', ' + str(top_label) + ')' if had_accident else ''}"
            )

    # Close last event and resources
    if open_event is not None:
        events.append(open_event)

    cap.release()
    if writer is not None:
        writer.release()

    progress.progress(100, text="Analyzing frames… 100%")
    elapsed = max(1e-6, time.time() - t0)
    processing_fps = processed_frames / elapsed

    # Build event rows + compute severity
    ev_out = []
    for i, e in enumerate(events, start=1):
        start_time = e["start_frame"] / fps
        duration_s = max(0.0, (e["end_frame"] - e["start_frame"] + 1) / max(1, fps))
        hit_frames = max(1, int(e.get("hit_frames", 1)))
        avg_area_ratio = float(e.get("area_sum", 0.0)) / hit_frames
        peak_conf = float(e.get("peak_conf", 0.0))
        severity_label = _classify_severity(duration_s, peak_conf, avg_area_ratio, DEFAULTS)

        ev_out.append({
            "event_id": i,
            "type": "accident",
            "subtype": e.get("type", "accident"),
            "start_time": round(float(start_time), 2),
            "severity": severity_label,
            "confidence": round(peak_conf, 3),
            "involved_track_ids": [],
            "peak_frame": int(e.get("peak_frame", e["start_frame"])),
        })

    events_df = pd.DataFrame([
        {
            "Start (s)": e["start_time"],
            "Type": e["type"],
            "Subtype": e["subtype"],
            "Severity": e["severity"],
            "Confidence": e["confidence"],
            "Involved IDs": "",
        }
        for e in ev_out
    ])

    # simple placeholder history & forecast (accident hits per bin)
    total_secs = int(total_frames / fps) + 1 if total_frames > 0 else (max(per_second_hits.keys() or [0]) + 1)
    bins = list(range(0, total_secs + 1, bin_size_sec))
    history_counts = []
    for i in range(1, len(bins)):
        start = bins[i - 1]
        end = bins[i]
        s = sum(per_second_hits.get(x, 0) for x in range(start, end))
        history_counts.append((end, s))
    if not history_counts:
        history_counts = [(bin_size_sec, 0)]
    ts_df = pd.DataFrame(history_counts, columns=["t_sec", "count"])

    steps = max(1, int((horizon_min * 60) / bin_size_sec))
    last_t = ts_df["t_sec"].iloc[-1]
    last_val = float(ts_df["count"].iloc[-1])
    fut_t = [last_t + (i + 1) * bin_size_sec for i in range(steps)]
    fut_v = [last_val + 0.05 * i for i in range(steps)]
    fc_df = pd.DataFrame({"t_sec": fut_t, "forecast": fut_v})

    # Prepare annotated video bytes if requested
    annotated_bytes = None
    annotated_name = None
    if export_video and tmp_video_path is not None and tmp_video_path.exists():
        annotated_name = f"annotated_{video_path.stem}.mp4"
        with open(tmp_video_path, "rb") as f:
            annotated_bytes = f.read()

    events_json = {
        "video": str(video_path.name),
        "fps": fps,
        "width": width,
        "height": height,
        "events": ev_out,
        "settings": {
            "conf_thresh": conf_thresh,
            "merge_gap_frames": merge_gap_frames,
            "bin_size_sec": bin_size_sec,
            "forecast_horizon_min": horizon_min,
            "device": device,
            "frame_skip": frame_skip,
            "imgsz": imgsz,
            "export_video": export_video,
            "sev_rules": {
                "sev_severe_min_dur_s": DEFAULTS["sev_severe_min_dur_s"],
                "sev_severe_min_area":  DEFAULTS["sev_severe_min_area"],
                "sev_severe_min_conf":  DEFAULTS["sev_severe_min_conf"],
                "sev_moderate_min_dur_s": DEFAULTS["sev_moderate_min_dur_s"],
                "sev_moderate_min_area":  DEFAULTS["sev_moderate_min_area"],
                "sev_moderate_min_conf":  DEFAULTS["sev_moderate_min_conf"],
            }
        },
        "class_names": CLASS_NAMES,
        "vehicle_class_ids": VEHICLE_CLASS_IDS,
        "accident_class_ids": ACCIDENT_CLASS_IDS,
        "allowed_classes": sorted(list(ALLOWED_CLASSES_SET)),
    }
    run_report = {
        "run_id": str(uuid.uuid4())[:8],
        "frames": total_frames,
        "processed_frames": processed_frames,
        "processing_fps": round(processing_fps, 2),
        "num_events": len(ev_out),
        "device": device,
        "model_weights": str(weights_path),
        "annotated_path": str(tmp_video_path) if tmp_video_path else None,
    }

    return {
        "run_id": run_report["run_id"],
        "events_df": events_df,
        "events_json": json.dumps(events_json, indent=2).encode(),
        "ts_df": ts_df,
        "fc_df": fc_df,
        "run_report": json.dumps(run_report, indent=2).encode(),
        "annotated_bytes": annotated_bytes,
        "annotated_name": annotated_name,
    }

# ---------------------------
# Sidebar — Upload/Select & Settings
# ---------------------------
with st.sidebar:
    st.header("Input & Controls")
    tab_in, tab_cfg = st.tabs(["Upload / Select", "Settings"])

    with tab_in:
        uploaded = st.file_uploader("Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
        text_path = st.text_input(
            "Or provide a local video path",
            value=str(DEFAULT_VIDEO_PATH),
            placeholder=str(DEFAULT_VIDEO_PATH),
        )
        weights_input = st.text_input(
            "Accident model weights (.pt)",
            value=str(DEFAULT_WEIGHTS_PATH),
            placeholder=str(DEFAULT_WEIGHTS_PATH),
        )
        export_video = st.checkbox("Export annotated video", True)
        export_logs = st.checkbox("Save JSON/CSV logs", True)
        run_btn = st.button("Run Analysis", use_container_width=True)

    with tab_cfg:
        st.caption("Analysis thresholds & runtime options")
        DEFAULTS["accident_score_thresh"] = st.slider("Accident score threshold", 0.1, 0.9, DEFAULTS["accident_score_thresh"], 0.01)
        DEFAULTS["merge_gap_frames"]      = st.slider("Event merge gap (frames)", 2, 60, DEFAULTS["merge_gap_frames"], 1)
        DEFAULTS["forecast_horizon_min"]  = st.select_slider("Forecast horizon (min)", options=[5, 10], value=DEFAULTS["forecast_horizon_min"])
        DEFAULTS["bin_size_sec"]          = st.select_slider("Bin size (sec)", options=[5, 10, 15], value=DEFAULTS["bin_size_sec"])
        DEFAULTS["device"]                = st.selectbox("Device", ["cpu", "cuda:0"], index=0)
        DEFAULTS["frame_skip"]            = st.slider("Frame skip (preview speed-up)", 0, 10, DEFAULTS["frame_skip"], 1)
        DEFAULTS["imgsz"]                 = st.select_slider("YOLO input size (imgsz)", options=[384, 512, 640, 768], value=DEFAULTS["imgsz"])

        st.divider()
        st.caption("Severity thresholds (rule-based)")
        DEFAULTS["sev_severe_min_dur_s"] = st.slider("Severe: min duration (s)", 0.5, 6.0, DEFAULTS["sev_severe_min_dur_s"], 0.1)
        DEFAULTS["sev_severe_min_area"]  = st.slider("Severe: min avg area ratio", 0.01, 0.30, DEFAULTS["sev_severe_min_area"], 0.01)
        DEFAULTS["sev_severe_min_conf"]  = st.slider("Severe: min peak conf", 0.30, 0.95, DEFAULTS["sev_severe_min_conf"], 0.01)

        DEFAULTS["sev_moderate_min_dur_s"] = st.slider("Moderate: min duration (s)", 0.3, 4.0, DEFAULTS["sev_moderate_min_dur_s"], 0.1)
        DEFAULTS["sev_moderate_min_area"]  = st.slider("Moderate: min avg area ratio", 0.01, 0.20, DEFAULTS["sev_moderate_min_area"], 0.01)
        DEFAULTS["sev_moderate_min_conf"]  = st.slider("Moderate: min peak conf", 0.30, 0.95, DEFAULTS["sev_moderate_min_conf"], 0.01)

# ---------------------------
# Main — Pages: Upload/Review/Downloads
# ---------------------------
st.title("Falcon Eye 360 — Offline Video")

colA, colB = st.columns([2.2, 1.0], gap="large")

with colA:
    st.subheader("Video Preview")
    video_src_path: Path | None = None

    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        video_src_path = Path(tmp.name)
        st.video(str(video_src_path))
    elif text_path.strip():
        p = resolve_path(text_path.strip())
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
            wpath = resolve_path(weights_input.strip()) if weights_input.strip() else None
            if (wpath is None) or (not wpath.exists()):
                st.error(f"Model weights not found: {weights_input}")
            else:
                with st.status("Running offline analysis…", expanded=True) as status:
                    st.write("Loading model & configuration…")
                    artifacts = _run_yolo_accident(
                        video_path=video_src_path,
                        weights_path=wpath,
                        conf_thresh=float(DEFAULTS["accident_score_thresh"]),
                        device=DEFAULTS["device"],
                        merge_gap_frames=int(DEFAULTS["merge_gap_frames"]),
                        bin_size_sec=int(DEFAULTS["bin_size_sec"]),
                        horizon_min=int(DEFAULTS["forecast_horizon_min"]),
                        frame_skip=int(DEFAULTS["frame_skip"]),
                        imgsz=int(DEFAULTS["imgsz"]),
                        export_video=bool(export_video),
                    )
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
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Events JSON", data=arts["events_json"], file_name="events.json", mime="application/json")
    with c2:
        merged_csv = pd.concat(
            [arts["ts_df"].assign(kind="history"), arts["fc_df"].assign(kind="forecast")],
            ignore_index=True
        ).to_csv(index=False).encode()
        st.download_button("Forecast CSV", data=merged_csv, file_name="forecast.csv", mime="text/csv")
    with c3:
        st.download_button("Run Report", data=arts["run_report"], file_name="run_report.json", mime="application/json")
    with c4:
        if arts.get("annotated_bytes") is not None:
            st.download_button("Annotated MP4", data=arts["annotated_bytes"], file_name=arts.get("annotated_name","annotated.mp4"), mime="video/mp4")
        else:
            st.button("Annotated MP4 (disabled)", disabled=True)

st.caption("v1.5: ALL classes enabled (0–9), accidents drive event/severity, annotated MP4 export.")
