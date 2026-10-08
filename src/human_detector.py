from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class HumanDetector:
    """Detects people in video frames using YOLOv8 and keeps track IDs when available."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", conf_threshold: float = 0.45):
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.device = device
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray):
        """Return detection dictionaries for people, including stable track IDs when ByteTrack is active."""
        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
                tracker="bytetrack.yaml",
            )
        except Exception:
            results = self.model(frame, conf=self.conf_threshold, verbose=False, device=self.device)

        detections = []
        if not results:
            return detections

        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = result.names.get(cls_id, str(cls_id))
            if label.lower() != "person":
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf.item())
            track_id = None
            if getattr(box, "id", None) is not None:
                track_id = int(box.id.item())

            detections.append({
                "bbox": tuple(map(int, xyxy)),
                "confidence": confidence,
                "label": label,
                "class_id": cls_id,
                "track_id": track_id,
            })

        return detections
