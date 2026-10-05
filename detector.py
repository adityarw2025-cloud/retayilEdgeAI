# detector.py
"""
YOLO Model Loading and Object Detection.
Performs pure object detection without tracking logic.
"""
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    """Standardized representation of a detected object."""
    bbox: tuple        # (x1, y1, x2, y2)
    confidence: float  # 0.0 - 1.0
    cls_id: int        # COCO class ID
    classname: str     # Human-readable class name
    center: tuple      # (cx, cy)
    is_person: bool    # True if person (class 0)


class ObjectDetector:
    """
    YOLO Model Loader & Object Detector.
    Encapsulates YOLO inference without any tracking logic.
    Supports configurable imgsz (e.g. 320 for 20+ FPS on CPU).
    """
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.40, imgsz=640):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = YOLO(model_path)
        self.names = self.model.names

    def detect(self, frame, imgsz=None):
        """
        Runs object detection on a frame.
        
        Args:
            frame: numpy BGR image.
            imgsz: int image size for inference (defaults to self.imgsz).
            
        Returns:
            detections: list of Detection objects.
            raw_boxes: Ultralytics Boxes object (passed to tracker.py).
        """
        target_imgsz = imgsz if imgsz is not None else self.imgsz
        results = self.model(frame, imgsz=target_imgsz, conf=self.conf_threshold, verbose=False)
        if not results or len(results) == 0:
            return [], None

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return [], boxes

        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            classname = self.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            is_person = (cls_id == 0 or classname.lower() == "person")

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                cls_id=cls_id,
                classname=classname,
                center=(cx, cy),
                is_person=is_person
            ))

        return detections, boxes

