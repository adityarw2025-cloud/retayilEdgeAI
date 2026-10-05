# yolo.py
"""
YOLO Person Tracking & Retail Analytics Pipeline.
Refactored to cleanly delegate to modular components:
- detector.py (YOLO model loading & inference)
- tracker.py (ByteTrack persistent tracking)
- zones.py (Store, Checkout, and Shelf zones)
- analytics.py (People counts, unique visitors, queue, dwell time)
- app.py (Dashboard UI & pipeline orchestrator)
"""
import app
from app import (
    VideoStream,
    process_frame,
    main,
    detector,
    tracker,
    analytics,
    CAM_SOURCE,
    MODEL_PATH,
    CONF_THRESHOLD,
    TRACKER_CONFIG as TRACKER_TYPE,
    QUEUE_THRESHOLD,
    DWELL_TIMEOUT as DWELL_TIMEOUT_SECONDS,
    PERSON_COLOR,
    PERSON_THICKNESS,
    DEFAULT_OBJECT_COLOR,
    OBJECT_THICKNESS,
    CLASS_COLORS,
)
from zones import (
    ZONES,
    QUEUE_ZONE_NAME,
    DWELL_ZONE_NAME,
    SHELF_ZONE_NAME,
    get_zone_pixels,
    resolve_all_zones,
    is_point_in_zone,
    get_zone_for_point
)
from analytics import RetailAnalytics, AnalyticsResult, StoreDwellTracker
from detector import ObjectDetector, Detection
from tracker import ObjectTracker, TrackedObject

# Backward-compatibility alias for model and dwell_tracker
model = detector.model
dwell_tracker = StoreDwellTracker()


if __name__ == "__main__":
    main()
