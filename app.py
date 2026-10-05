# app.py
"""
RetailSense - AI Vision PoC
Real-time Customer Intelligence, Queue Management & Dwell Analytics Dashboard.
Features:
- Configurable Store Area, Checkout Area, and Shelf Area zones
- Retail Customer Simulation Mode (demonstrates customer count, queue count, dwell time, occupancy)
- Clear "Future Retail Product Detection" architecture documentation
- Pure modular pipeline (detector -> tracker -> analytics -> zones)
"""
import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st

from detector import ObjectDetector
from tracker import ObjectTracker
from zones import (
    ZONES,
    PRIMARY_ZONES,
    resolve_all_zones,
    QUEUE_ZONE_NAME,
    DWELL_ZONE_NAME,
    SHELF_ZONE_NAME
)
from analytics import RetailAnalytics, StoreDwellTracker
from simulation import RetailSimulator

# =====================================================================
# Standalone Pipeline Constants & Helpers (Non-Streamlit / CLI Mode)
# =====================================================================
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.40
TRACKER_CONFIG = "bytetrack.yaml"
QUEUE_THRESHOLD = 5
DWELL_TIMEOUT = 2.0
CAM_SOURCE = 0
PERSON_THICKNESS = 2
OBJECT_THICKNESS = 1

# Pre-initialized instances for direct script execution or testing
detector = ObjectDetector(model_path=MODEL_PATH, conf_threshold=CONF_THRESHOLD)
tracker = ObjectTracker(config_path=TRACKER_CONFIG, names=detector.names)
analytics = RetailAnalytics(
    queue_zone=QUEUE_ZONE_NAME,
    queue_threshold=QUEUE_THRESHOLD,
    dwell_zone=DWELL_ZONE_NAME,
    dwell_timeout=DWELL_TIMEOUT
)

class VideoStream:
    """Wrapper around cv2.VideoCapture for modular access."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
    def read(self):
        return self.cap.read()
    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


# =====================================================================
# Streamlit Page Configuration
# =====================================================================
st.set_page_config(
    page_title="RetailSense - AI Vision PoC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Professional Modern Theme & Styling (RetailSense)
# =====================================================================
st.markdown("""
<style>
    /* Global Typography & Spacing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        background-color: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(8px);
        z-index: 99;
    }
    
    /* Header & Branding */
    .rs-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 12px;
        margin-bottom: 14px;
    }
    .rs-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .rs-logo-icon {
        font-size: 1.8rem;
        line-height: 1;
    }
    .rs-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
    }
    .rs-title-accent {
        color: #38bdf8;
    }
    .rs-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 4px;
    }
    
    /* AI Vision PoC Status Indicator Badge */
    .rs-poc-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .rs-pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px #10b981;
    }
    .rs-badge-sep {
        color: #475569;
    }
    .rs-status-text {
        color: #38bdf8;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: left;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .card-people { border-top: 3px solid #38bdf8; }
    .card-queue-normal { border-top: 3px solid #f59e0b; }
    .card-queue-alert { 
        border-top: 3px solid #ef4444; 
        background: linear-gradient(180deg, #2d1215 0%, #15080a 100%);
        border-color: #7f1d1d;
    }
    .card-dwell { border-top: 3px solid #34d399; }
    .card-fps { border-top: 3px solid #a78bfa; }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .metric-lbl {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.06em;
    }
    .metric-icon {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .metric-val-alert {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f87171;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Professional Alert Banner */
    .rs-alert-box {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 8px;
        padding: 9px 16px;
        margin-bottom: 14px;
        font-size: 0.88rem;
    }
    .rs-alert-normal {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }
    .rs-alert-critical {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
    }
    .rs-alert-content {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .rs-alert-tag {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .rs-tag-ok {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
    }
    .rs-tag-alert {
        background: rgba(239, 68, 68, 0.3);
        color: #f87171;
    }

    /* Video Header Bar */
    .video-header-bar {
        background: #0f172a;
        border: 1px solid #334155;
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 8px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
    }
    .video-title-live {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #e2e8f0;
    }
    .live-indicator {
        width: 8px;
        height: 8px;
        background-color: #ef4444;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px #ef4444;
    }

    /* Side breakdown cards */
    .side-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .side-card-title {
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .zone-stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #1e293b;
        font-size: 0.84rem;
    }
    .zone-stat-row:last-child {
        border-bottom: none;
    }
    .zone-name-label {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #cbd5e1;
    }
    .zone-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .zone-count-badge {
        background: #1e293b;
        color: #f8fafc;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* Future Product Detection Card */
    .future-section {
        background: linear-gradient(180deg, #111827 0%, #0b0f19 100%);
        border: 1px solid #1e293b;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 20px;
    }
    .future-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .future-badge {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }
    .future-p {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.45;
        margin-bottom: 10px;
    }
    .future-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 12px;
        margin-top: 10px;
    }
    .future-item {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 12px;
    }
    .future-item-title {
        font-size: 0.80rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .future-item-desc {
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.35;
    }
</style>
""", unsafe_allow_html=True)

# Visual styling constants for OpenCV annotation
PERSON_COLOR = (0, 0, 255)            # Red for Person (BGR)
DEFAULT_OBJECT_COLOR = (0, 230, 100)  # Green for other objects
CLASS_COLORS = {
    'chair': (255, 140, 0),
    'couch': (255, 140, 0),
    'laptop': (255, 0, 255),
    'bottle': (0, 230, 255),
    'cell phone': (200, 200, 0),
    'backpack': (180, 105, 255),
    'book': (255, 215, 0),
    'dining table': (160, 160, 160),
    'cup': (0, 200, 255),
}


# =====================================================================
# Model & Pipeline Initialization
# =====================================================================
@st.cache_resource
def load_detector():
    """Initializes and caches the YOLO object detector once globally."""
    return ObjectDetector(model_path="yolov8n.pt", conf_threshold=0.40, imgsz=320)


def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_len=14):
    """Draws a high-tech corner-bracket bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    cl = min(corner_len, w // 3, h // 3)

    # Base rectangle outline (thin 1px)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    # Corner brackets (prominent)
    # Top-Left Corner
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness)
    # Top-Right Corner
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, thickness)
    # Bottom-Left Corner
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness)
    # Bottom-Right Corner
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness)


def annotate_frame(frame, tracked_objects, analytics_res, resolved_zones):
    """
    Renders the computer-vision visualization on the frame:
    - Store Area, Shelf Area, and Checkout Area boundaries with clean HUD tags
    - Person red bounding boxes with corner brackets, yellow center point dot, persistent ID, and dwell tags
    - Standard bounding boxes for other objects
    - Alert coloring on Checkout Area if queue threshold is exceeded
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Draw Configurable Zone Rectangles and Headers (Store Area, Shelf Area, Checkout Area)
    for name in PRIMARY_ZONES:
        if name not in resolved_zones or name not in ZONES:
            continue
        info = ZONES[name]
        zx1, zy1, zx2, zy2 = resolved_zones[name]
        is_checkout = (name in ("Checkout Area", "Checkout Zone"))
        count = analytics_res.zone_occupancy.get(name, 0)

        if is_checkout and analytics_res.is_queue_alert:
            z_color = (0, 0, 255)  # Alert Red
            z_label = f"CHECKOUT AREA | Queue: {analytics_res.queue_count} [ALERT!]"
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), z_color, 3)
            tag_bg = z_color
            txt_color = (255, 255, 255)
        elif is_checkout:
            z_color = info["color"]  # Orange
            z_label = f"CHECKOUT AREA | Queue: {analytics_res.queue_count}"
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), z_color, 2)
            tag_bg = (20, 20, 30)
            txt_color = z_color
        elif name in ("Store Area", "Store Zone"):
            z_color = info["color"]  # Cyan
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), z_color, 2)
            if analytics_res.avg_dwell > 0:
                z_label = f"STORE AREA | Shoppers: {count} | Avg Dwell: {analytics_res.avg_dwell:.1f}s"
            else:
                z_label = f"STORE AREA | Shoppers: {count}"
            tag_bg = (20, 20, 30)
            txt_color = z_color
        else:
            z_color = info["color"]  # Green
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), z_color, 2)
            z_label = f"SHELF AREA | Shoppers: {count}"
            tag_bg = (20, 20, 30)
            txt_color = z_color

        # Sleek HUD Tag above zone
        (zw, zh), _ = cv2.getTextSize(z_label, font, 0.48, 1)
        tag_top = max(zy1 - zh - 8, 8)
        cv2.rectangle(annotated, (zx1, tag_top), (zx1 + zw + 12, tag_top + zh + 7), tag_bg, cv2.FILLED)
        cv2.rectangle(annotated, (zx1, tag_top), (zx1 + zw + 12, tag_top + zh + 7), z_color, 1)
        cv2.putText(annotated, z_label, (zx1 + 6, tag_top + zh + 2), font, 0.48, txt_color, 1, cv2.LINE_AA)

    # 2. Draw Tracked Objects
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        conf_pct = int(obj.confidence * 100)

        if obj.is_person:
            tid = obj.track_id
            cx, cy = obj.center

            # Label text
            if tid is not None and tid in analytics_res.current_dwells:
                dwell_s = analytics_res.current_dwells[tid]
                label = f"Person ID: {tid} ({dwell_s:.1f}s)"
            elif tid is not None:
                label = f"Person ID: {tid}"
            else:
                label = f"Person {conf_pct}%"

            # Red corner-bracket box & yellow center dot with black outline
            draw_corner_rect(annotated, (x1, y1), (x2, y2), PERSON_COLOR, thickness=2, corner_len=14)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), cv2.FILLED)

            # Label banner above box
            (w, h), _ = cv2.getTextSize(label, font, 0.50, 1)
            top_y = max(y1, h + 10)
            cv2.rectangle(annotated, (x1, top_y - h - 5), (x1 + w + 8, top_y + 2), PERSON_COLOR, cv2.FILLED)
            cv2.putText(annotated, label, (x1 + 4, top_y - 2), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            # Dwell badge below box for Store Area occupants
            if tid is not None and tid in analytics_res.current_dwells:
                badge = f"Dwell: {analytics_res.current_dwells[tid]:.1f}s"
                (bw, bh), _ = cv2.getTextSize(badge, font, 0.45, 1)
                b_top = min(y2 + bh + 4, height - 5)
                cv2.rectangle(annotated, (x1, b_top - bh - 2), (x1 + bw + 6, b_top + 3), (255, 191, 0), cv2.FILLED)
                cv2.putText(annotated, badge, (x1 + 3, b_top), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Other COCO objects (chairs, laptops, bottles, etc.)
            obj_color = CLASS_COLORS.get(obj.classname, DEFAULT_OBJECT_COLOR)
            label = f"{obj.classname} {conf_pct}%"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), obj_color, 1)
            (w, h), _ = cv2.getTextSize(label, font, 0.45, 1)
            top_y = max(y1, h + 8)
            cv2.rectangle(annotated, (x1, top_y - h - 4), (x1 + w + 6, top_y + 2), (20, 20, 30), cv2.FILLED)
            cv2.rectangle(annotated, (x1, top_y - h - 4), (x1 + w + 6, top_y + 2), obj_color, 1)
            cv2.putText(annotated, label, (x1 + 3, top_y - 2), font, 0.45, obj_color, 1, cv2.LINE_AA)

    # Convert BGR to RGB for Streamlit rendering
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def process_frame(frame, cam_label="Camera", fps=0.0, tracker_instance=None, dwell_tracker=None, **kwargs):
    """Processes a frame through detection, tracking, zones, and analytics (returns 7-tuple for OpenCV/CLI)."""
    h, w = frame.shape[:2]
    resolved_zones = resolve_all_zones(ZONES, w, h)
    detections, raw_boxes = detector.detect(frame)

    active_obj_tracker = tracker
    active_analytics = analytics
    if tracker_instance is not None:
        if isinstance(tracker_instance, StoreDwellTracker):
            active_analytics = tracker_instance
        else:
            active_obj_tracker = tracker_instance
    if dwell_tracker is not None:
        active_analytics = dwell_tracker

    tracked_objects = active_obj_tracker.update(raw_boxes, frame, class_names=detector.names)
    res = active_analytics.update(tracked_objects, resolved_zones)
    annotated_rgb = annotate_frame(frame, tracked_objects, res, resolved_zones)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    active_ids = [obj.track_id for obj in tracked_objects if obj.is_person and obj.track_id is not None]
    return annotated_bgr, active_ids, len(active_ids), res.zone_occupancy, res.queue_count, res.is_queue_alert, res.avg_dwell


# =====================================================================
# Main Streamlit Application
# =====================================================================
def main():
    # -------------------------------------------------------------
    # Header Section with AI Vision PoC Status Indicator
    # -------------------------------------------------------------
    st.markdown("""
    <div class="rs-header">
        <div class="rs-title-group">
            <div class="rs-brand">
                <span class="rs-logo-icon">⚡</span>
                <span class="rs-title">Retail<span class="rs-title-accent">Sense</span></span>
            </div>
            <div class="rs-subtitle">Edge Computer-Vision Intelligence & Customer Dynamics (PoC)</div>
        </div>
        <div class="rs-poc-badge">
            <span class="rs-pulse-dot"></span>
            <span>AI VISION POC</span>
            <span class="rs-badge-sep">•</span>
            <span class="rs-status-text">ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Sidebar Configuration (RetailSense Console)
    # -------------------------------------------------------------
    st.sidebar.markdown("### ⚡ RetailSense Console")
    st.sidebar.caption("System Configuration & Mode Controls")

    st.sidebar.markdown("#### 📡 Stream Source")
    input_source = st.sidebar.selectbox(
        "Input Source",
        options=["Retail Simulation Mode", "Webcam", "Upload Video File", "RTSP / IP Camera", "Synthetic Benchmark Scene"],
        index=0
    )

    # Initialize Simulator state
    if "simulator" not in st.session_state:
        st.session_state["simulator"] = RetailSimulator(width=960, height=600)
    sim = st.session_state["simulator"]

    video_path = None
    if input_source == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
    elif input_source == "RTSP / IP Camera":
        st.sidebar.markdown("""
        <div style="font-size:0.75rem; color:#94a3b8; background:#0f172a; border:1px solid #334155; border-radius:6px; padding:8px; margin-bottom:8px;">
            📱 <strong>IP Webcam URL Examples:</strong><br>
            • Android IP Webcam: <code>http://&lt;PHONE_IP&gt;:8080/video</code><br>
            • DroidCam: <code>http://&lt;PHONE_IP&gt;:4747/video</code><br>
            • RTSP Stream: <code>rtsp://&lt;user&gt;:&lt;pass&gt;@&lt;IP&gt;:554/h264</code>
        </div>
        """, unsafe_allow_html=True)
        video_path = st.sidebar.text_input(
            "Stream URL", 
            value="http://192.168.1.100:8080/video",
            help="Enter the IP webcam video stream URL shown in your phone's camera app."
        )
    elif input_source == "Synthetic Benchmark Scene":
        video_path = r"venv\Lib\site-packages\ultralytics\assets\bus.jpg"

    # Interactive Simulation Controls
    if input_source == "Retail Simulation Mode":
        st.sidebar.markdown("#### 🎮 Simulation Controls")
        sim_speed = st.sidebar.select_slider("Simulation Speed", options=["0.5x", "1.0x", "1.5x", "2.0x"], value="1.0x")
        speed_val = float(sim_speed.replace("x", ""))
        sim.speed_multiplier = speed_val

        col_btn1, col_btn2 = st.sidebar.columns(2)
        with col_btn1:
            if st.button("🚨 Rush Queue", help="Trigger 4+ customers entering Checkout Area to test Queue Alert"):
                sim.trigger_rush_hour()
                st.sidebar.warning("Rush hour traffic sent to Checkout!")
        with col_btn2:
            if st.button("➕ Add Shopper", help="Spawn a new simulated customer"):
                sim.spawn_customer()

        if st.sidebar.button("🔄 Reset Shoppers"):
            sim.reset()
            st.sidebar.info("Simulation reset.")

    st.sidebar.markdown("#### ⚡ Performance Mode")
    perf_mode = st.sidebar.select_slider(
        "Inference Resolution",
        options=["Ultra Fast (320p)", "Balanced (480p)", "High Precision (640p)"],
        value="Ultra Fast (320p)",
        help="320p runs at 20-25+ FPS on CPU. 640p runs at 4-5 FPS on CPU."
    )
    imgsz_map = {
        "Ultra Fast (320p)": 320,
        "Balanced (480p)": 480,
        "High Precision (640p)": 640
    }
    selected_imgsz = imgsz_map[perf_mode]

    st.sidebar.markdown("#### 🎯 Intelligence Thresholds")
    queue_threshold = st.sidebar.slider("Queue Alert Threshold", min_value=1, max_value=15, value=5)
    conf_threshold = st.sidebar.slider("Detection Confidence", min_value=0.20, max_value=0.85, value=0.40, step=0.05)
    dwell_timeout = st.sidebar.slider("Dwell Timeout (seconds)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    st.sidebar.markdown("#### ⚙️ Runtime Controls")
    is_running = st.sidebar.toggle("▶️ Run Live Processing", value=True)

    if st.sidebar.button("🔄 Reset Analytics Counters"):
        if "analytics_engine" in st.session_state:
            st.session_state["analytics_engine"].reset()
        st.sidebar.success("Analytics state reset!")

    # -------------------------------------------------------------
    # Initialize Core Pipeline Components
    # -------------------------------------------------------------
    detector = load_detector()
    detector.conf_threshold = conf_threshold
    detector.imgsz = selected_imgsz

    if "tracker_engine" not in st.session_state:
        st.session_state["tracker_engine"] = ObjectTracker(config_path="bytetrack.yaml", names=detector.names)
    tracker = st.session_state["tracker_engine"]

    if "analytics_engine" not in st.session_state:
        st.session_state["analytics_engine"] = RetailAnalytics(
            queue_zone=QUEUE_ZONE_NAME,
            queue_threshold=queue_threshold,
            dwell_zone=DWELL_ZONE_NAME,
            dwell_timeout=dwell_timeout
        )
    analytics = st.session_state["analytics_engine"]
    analytics.queue_threshold = queue_threshold
    analytics.dwell_timeout = dwell_timeout

    # -------------------------------------------------------------
    # Top Metrics Section (4 Modern KPI Cards)
    # -------------------------------------------------------------
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        ph_people = st.empty()
    with col_kpi2:
        ph_queue = st.empty()
    with col_kpi3:
        ph_dwell = st.empty()
    with col_kpi4:
        ph_fps = st.empty()

    # Initial placeholder render for KPIs
    ph_people.markdown("""
    <div class="metric-card card-people">
        <div class="metric-header"><span class="metric-lbl">People Detected</span><span class="metric-icon">👥</span></div>
        <div class="metric-val">0</div>
        <div class="metric-sub">Active in scene • ByteTrack</div>
    </div>
    """, unsafe_allow_html=True)

    ph_queue.markdown(f"""
    <div class="metric-card card-queue-normal">
        <div class="metric-header"><span class="metric-lbl">Queue Count</span><span class="metric-icon">🚶</span></div>
        <div class="metric-val">0</div>
        <div class="metric-sub">Checkout Area • Limit: {queue_threshold}</div>
    </div>
    """, unsafe_allow_html=True)

    ph_dwell.markdown("""
    <div class="metric-card card-dwell">
        <div class="metric-header"><span class="metric-lbl">Average Dwell Time</span><span class="metric-icon">⏱️</span></div>
        <div class="metric-val">0.0s</div>
        <div class="metric-sub">Store Area active shoppers</div>
    </div>
    """, unsafe_allow_html=True)

    ph_fps.markdown("""
    <div class="metric-card card-fps">
        <div class="metric-header"><span class="metric-lbl">Throughput</span><span class="metric-icon">⚡</span></div>
        <div class="metric-val">0.0</div>
        <div class="metric-sub">Real-time FPS • YOLOv8n</div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # High-Contrast Alert Section
    # -------------------------------------------------------------
    alert_placeholder = st.empty()
    alert_placeholder.markdown(f"""
    <div class="rs-alert-box rs-alert-normal">
        <div class="rs-alert-content">
            <span style="font-size: 1.1rem;">✅</span>
            <span><strong>Checkout Area Normal:</strong> 0 customers in queue (Alert threshold: {queue_threshold}).</span>
        </div>
        <span class="rs-alert-tag rs-tag-ok">Optimal</span>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Main Video Display & Analytics Breakdown Section
    # -------------------------------------------------------------
    main_col, side_info_col = st.columns([7.6, 2.4])

    with main_col:
        stream_tag = "RETAIL SIMULATION • STORE OPERATIONS" if input_source == "Retail Simulation Mode" else "LIVE CV STREAM • STORE OPERATIONS"
        st.markdown(f"""
        <div class="video-header-bar">
            <div class="video-title-live">
                <span class="live-indicator"></span>
                <span>{stream_tag}</span>
            </div>
            <span>MODEL: YOLOv8n + ByteTrack</span>
        </div>
        """, unsafe_allow_html=True)
        video_placeholder = st.empty()

    with side_info_col:
        st.markdown("""
        <div class="side-card">
            <div class="side-card-title">📍 Zone Occupancy</div>
        </div>
        """, unsafe_allow_html=True)
        ph_zone_stats = st.empty()

        st.markdown("""
        <div class="side-card" style="margin-top: 10px;">
            <div class="side-card-title">⏱️ Active Store Dwellers</div>
        </div>
        """, unsafe_allow_html=True)
        ph_dwell_table = st.empty()

    # -------------------------------------------------------------
    # Future Retail Product Detection Architecture Section
    # -------------------------------------------------------------
    st.markdown("""
    <div class="future-section">
        <div class="future-title">
            <span>📦 Future Retail Product Detection (Shelf Area Roadmap)</span>
            <span class="future-badge">Architecture & Vision Roadmap</span>
        </div>
        <div class="future-p">
            <strong>⚠️ PoC Scope Clarification:</strong> The current system monitors <em>customer foot traffic, dwell duration, and customer density</em> inside the <strong>Shelf Area</strong> using the general-purpose pretrained YOLOv8 person model. It does <strong>not</strong> claim to identify specific shelf items or inventory levels with COCO weights.
        </div>
        <div class="future-grid">
            <div class="future-item">
                <div class="future-item-title">1. Retail-Specific Annotated Dataset</div>
                <div class="future-item-desc">Production product detection requires densely labeled retail datasets such as SKU-110K, RP2K, or custom supermarket product annotations covering thousands of packaged retail goods.</div>
            </div>
            <div class="future-item">
                <div class="future-item-title">2. Fine-Tuned Detection Model</div>
                <div class="future-item-desc">A specialized, fine-tuned YOLOv8-OBB (Oriented Bounding Boxes) or multi-task vision transformer will detect individual product SKUs, brand facings, and shelf item counts.</div>
            </div>
            <div class="future-item">
                <div class="future-item-title">3. Out-of-Stock & Planogram Fusion</div>
                <div class="future-item-desc">By fusing customer dwell analytics in the Shelf Area with fine-grained product bounding boxes, the system will identify out-of-stock (OOS) gaps, planogram compliance, and pick-and-go behavior.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_running:
        video_placeholder.info("Processing paused. Toggle 'Run Live Processing' in the sidebar to start live feed.")
        return

    # -------------------------------------------------------------
    # Video Ingestion Setup
    # -------------------------------------------------------------
    cap = None
    single_image_mode = False
    simulation_mode = (input_source == "Retail Simulation Mode")

    if simulation_mode:
        pass  # Handled by sim.get_frame()
    elif input_source == "Webcam":
        cap = cv2.VideoCapture(0)
    elif input_source == "Upload Video File":
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
        else:
            video_placeholder.warning("Please upload a video file in the sidebar to begin processing.")
            return
    elif input_source == "RTSP / IP Camera":
        if video_path and video_path.strip():
            cap = cv2.VideoCapture(video_path.strip())
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            video_placeholder.warning("Please enter your IP Webcam or RTSP Stream URL in the sidebar.")
            return
    elif input_source == "Synthetic Benchmark Scene":
        single_image_mode = True

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    prev_alert_state = None

    try:
        while is_running:
            frame_start = time.time()
            if simulation_mode:
                frame = sim.get_frame()
            elif single_image_mode:
                frame = cv2.imread(video_path)
                if frame is None:
                    video_placeholder.error(f"Could not load benchmark image: {video_path}")
                    break
            else:
                if cap is None or not cap.isOpened():
                    video_placeholder.error("Cannot open video stream or webcam. Check camera connection or URL.")
                    break
                ret, frame = cap.read()
                if not ret:
                    if input_source == "Upload Video File":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        video_placeholder.warning("No video frame received. Waiting for feed...")
                        time.sleep(0.5)
                        continue

            # Downscale high-resolution frames (e.g. 1080p/4K from phone IP webcams) to max width 960
            if frame is not None and frame.shape[1] > 960:
                scale = 960.0 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

            # Calculate live FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)
            prev_time = curr_time

            # -------------------------------------------------------------
            # Core Pipeline: detector -> tracker -> analytics -> zones
            # -------------------------------------------------------------
            height, width = frame.shape[:2]
            resolved_zones = resolve_all_zones(ZONES, width, height)

            # 1. YOLO Object Detection (Pure inference, no tracking)
            detections, raw_boxes = detector.detect(frame)

            # 2. ByteTrack Persistent Multi-Object Tracking
            tracked_objects = tracker.update(raw_boxes, frame, class_names=detector.names)

            # 3. Retail Analytics (People count, queue count, dwell time)
            analytics_res = analytics.update(tracked_objects, resolved_zones, current_time=curr_time)

            # 4. Annotate Video Frame
            annotated_rgb = annotate_frame(frame, tracked_objects, analytics_res, resolved_zones)

            # -------------------------------------------------------------
            # Update Dashboard UI
            # -------------------------------------------------------------
            frame_count += 1
            # Real-time video frame updated every frame
            video_placeholder.image(annotated_rgb, use_container_width=True)

            # Secondary metric updates throttled to every 3 frames or immediate on alert status transition
            alert_state_changed = (analytics_res.is_queue_alert != prev_alert_state)
            if frame_count % 3 == 0 or alert_state_changed:
                prev_alert_state = analytics_res.is_queue_alert

                # Top KPI 1: People
                ph_people.markdown(f"""
                <div class="metric-card card-people">
                    <div class="metric-header"><span class="metric-lbl">People Detected</span><span class="metric-icon">👥</span></div>
                    <div class="metric-val">{analytics_res.num_active_people}</div>
                    <div class="metric-sub">Total Visitors: {analytics_res.total_unique_visitors}</div>
                </div>
                """, unsafe_allow_html=True)

                # Top KPI 2: Queue (Card style dynamically changes on alert)
                queue_card_class = "card-queue-alert" if analytics_res.is_queue_alert else "card-queue-normal"
                queue_val_class = "metric-val-alert" if analytics_res.is_queue_alert else "metric-val"
                ph_queue.markdown(f"""
                <div class="metric-card {queue_card_class}">
                    <div class="metric-header"><span class="metric-lbl">Queue Count</span><span class="metric-icon">🚶</span></div>
                    <div class="{queue_val_class}">{analytics_res.queue_count}</div>
                    <div class="metric-sub">Checkout Area • Limit: {queue_threshold}</div>
                </div>
                """, unsafe_allow_html=True)

                # Top KPI 3: Dwell
                store_occupants = analytics_res.zone_occupancy.get("Store Area", analytics_res.zone_occupancy.get("Store Zone", 0))
                ph_dwell.markdown(f"""
                <div class="metric-card card-dwell">
                    <div class="metric-header"><span class="metric-lbl">Average Dwell Time</span><span class="metric-icon">⏱️</span></div>
                    <div class="metric-val">{analytics_res.avg_dwell:.1f}s</div>
                    <div class="metric-sub">Store Area ({store_occupants} active)</div>
                </div>
                """, unsafe_allow_html=True)

                # Top KPI 4: FPS
                ph_fps.markdown(f"""
                <div class="metric-card card-fps">
                    <div class="metric-header"><span class="metric-lbl">Throughput</span><span class="metric-icon">⚡</span></div>
                    <div class="metric-val">{fps:.1f}</div>
                    <div class="metric-sub">Real-time FPS • YOLOv8n ({detector.imgsz}p)</div>
                </div>
                """, unsafe_allow_html=True)

                # Alert Section
                if analytics_res.is_queue_alert:
                    alert_placeholder.markdown(f"""
                    <div class="rs-alert-box rs-alert-critical">
                        <div class="rs-alert-content">
                            <span style="font-size: 1.2rem;">🚨</span>
                            <span><strong>QUEUE OVERFLOW ALERT:</strong> {analytics_res.queue_count} people in Checkout Area exceeds limit of {queue_threshold}! Open additional lane immediately.</span>
                        </div>
                        <span class="rs-alert-tag rs-tag-alert">Action Required</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    alert_placeholder.markdown(f"""
                    <div class="rs-alert-box rs-alert-normal">
                        <div class="rs-alert-content">
                            <span style="font-size: 1.1rem;">✅</span>
                            <span><strong>Checkout Area Normal:</strong> {analytics_res.queue_count} customer(s) in queue (Alert threshold: {queue_threshold}).</span>
                        </div>
                        <span class="rs-alert-tag rs-tag-ok">Optimal</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Side Panel 1: Zone Occupancy (Store Area, Shelf Area, Checkout Area)
                store_count = analytics_res.zone_occupancy.get("Store Area", analytics_res.zone_occupancy.get("Store Zone", 0))
                shelf_count = analytics_res.zone_occupancy.get("Shelf Area", analytics_res.zone_occupancy.get("Shelf Zone", 0))
                queue_count = analytics_res.zone_occupancy.get("Checkout Area", analytics_res.zone_occupancy.get("Checkout Zone", 0))
                ph_zone_stats.markdown(f"""
                <div class="side-card">
                    <div class="zone-stat-row">
                        <span class="zone-name-label"><span class="zone-dot" style="background:#38bdf8;"></span> Store Area</span>
                        <span class="zone-count-badge">{store_count}</span>
                    </div>
                    <div class="zone-stat-row">
                        <span class="zone-name-label"><span class="zone-dot" style="background:#10b981;"></span> Shelf Area</span>
                        <span class="zone-count-badge">{shelf_count}</span>
                    </div>
                    <div class="zone-stat-row">
                        <span class="zone-name-label"><span class="zone-dot" style="background:#f59e0b;"></span> Checkout Area</span>
                        <span class="zone-count-badge">{queue_count}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Side Panel 2: Active Store Dwellers
                if analytics_res.current_dwells:
                    dwell_rows = "".join([
                        f"""<div class="zone-stat-row">
                            <span class="zone-name-label">👤 Person ID #{tid}</span>
                            <span class="zone-count-badge" style="background:#0369a1;">{sec:.1f}s</span>
                        </div>"""
                        for tid, sec in analytics_res.current_dwells.items()
                    ])
                    ph_dwell_table.markdown(f'<div class="side-card">{dwell_rows}</div>', unsafe_allow_html=True)
                else:
                    ph_dwell_table.markdown("""
                    <div class="side-card" style="color:#64748b; font-size:0.80rem; text-align:center; padding:12px;">
                        No customers currently in Store Area.
                    </div>
                    """, unsafe_allow_html=True)

            # Smooth dynamic pacing only for simulation and benchmark modes (not for live cameras)
            if simulation_mode or single_image_mode:
                elapsed_cycle = time.time() - frame_start
                target_dt = 1.0 / 30.0
                if elapsed_cycle < target_dt:
                    time.sleep(target_dt - elapsed_cycle)

    finally:
        if cap and hasattr(cap, "release") and cap.isOpened():
            cap.release()


if __name__ == "__main__":
    main()
