# tracker.py
"""
ByteTrack Multi-Object Tracking & Persistent Identity Management.
Assigns and maintains persistent tracking IDs across consecutive frames.
"""
import os
import yaml
from types import SimpleNamespace
from dataclasses import dataclass
from ultralytics.trackers.byte_tracker import BYTETracker


@dataclass
class TrackedObject:
    """Standardized representation of a tracked object with persistent ID."""
    track_id: int | None  # Persistent tracking ID (None if unconfirmed/unassigned)
    bbox: tuple           # (x1, y1, x2, y2)
    confidence: float     # 0.0 - 1.0
    cls_id: int           # COCO class ID
    classname: str        # Class name
    center: tuple         # (cx, cy)
    is_person: bool       # True if person (class 0)


class ObjectTracker:
    """
    ByteTrack Tracker for persistent tracking IDs.
    Separates tracking state from model inference.
    """
    def __init__(self, config_path="bytetrack.yaml", names=None):
        self.config_path = config_path
        self.names = names or {0: "person"}
        self.tracker = self._init_tracker()

    def _init_tracker(self):
        cfg = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            # Sane defaults matching optimized bytetrack.yaml
            cfg = {
                "tracker_type": "bytetrack",
                "track_high_thresh": 0.40,
                "track_low_thresh": 0.10,
                "new_track_thresh": 0.55,
                "track_buffer": 90,
                "match_thresh": 0.50,
                "fuse_score": True
            }
        args = SimpleNamespace(**cfg)
        return BYTETracker(args)

    def update(self, raw_boxes, frame, class_names=None):
        """
        Updates ByteTrack with detections from detector.py.
        
        Args:
            raw_boxes: Ultralytics Boxes object from detector.
            frame: numpy BGR image.
            class_names: optional dict mapping cls_id -> classname.
            
        Returns:
            list of TrackedObject instances.
        """
        if class_names:
            self.names = class_names

        if raw_boxes is None or len(raw_boxes) == 0:
            return []

        # BYTETracker.update returns numpy array:
        # Each row: [x1, y1, x2, y2, track_id, conf, cls_id, det_idx]
        tracks = self.tracker.update(raw_boxes, frame)
        if tracks is None or len(tracks) == 0:
            # Fallback for unconfirmed initial frame: emit detected objects without track_id
            tracked_objects = []
            for box in raw_boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                classname = self.names.get(cls_id, str(cls_id))
                is_p = (cls_id == 0 or classname.lower() == "person")
                tracked_objects.append(TrackedObject(
                    track_id=None,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    cls_id=cls_id,
                    classname=classname,
                    center=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    is_person=is_p
                ))
            return tracked_objects

        tracked_objects = []
        for trk in tracks:
            x1, y1, x2, y2 = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3])
            track_id = int(trk[4])
            conf = float(trk[5])
            cls_id = int(trk[6])
            classname = self.names.get(cls_id, str(cls_id))
            is_p = (cls_id == 0 or classname.lower() == "person")

            tracked_objects.append(TrackedObject(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                cls_id=cls_id,
                classname=classname,
                center=(int((x1 + x2) / 2), int((y1 + y2) / 2)),
                is_person=is_p
            ))

        return tracked_objects

    def reset(self):
        """Resets tracker state."""
        self.tracker = self._init_tracker()

