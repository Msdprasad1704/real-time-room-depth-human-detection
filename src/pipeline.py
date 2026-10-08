from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np
import torch

from src.depth_model import MonoDepthEstimator
from src.heatmap import create_depth_heatmap
from src.human_detector import HumanDetector


class DepthHumanPipeline:
    """Runs the full pipeline: detect humans, estimate depth, and display live monitoring output."""

    def __init__(self, device: str = "auto", conf_threshold: float = 0.45):
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = HumanDetector(device=self.device, conf_threshold=conf_threshold)
        self.depth_estimator = MonoDepthEstimator(device=self.device)
        self.show_heatmap = True
        self.show_depth_info = True
        self.fps = 0.0
        self._track_positions: dict[int, tuple[float, float]] = {}
        self._next_person_id = 1
        self.last_detections: list[dict[str, Any]] = []
        self.last_stats: dict[str, Any] = {}

    def toggle_heatmap(self) -> bool:
        self.show_heatmap = not self.show_heatmap
        return self.show_heatmap

    def toggle_depth_info(self) -> bool:
        self.show_depth_info = not self.show_depth_info
        return self.show_depth_info

    @staticmethod
    def _depth_zone(relative_depth: float) -> str:
        if relative_depth >= 0.67:
            return "NEAR"
        if relative_depth >= 0.34:
            return "MEDIUM"
        return "FAR"

    @staticmethod
    def _room_occupancy(count: int) -> str:
        if count == 0:
            return "LOW"
        if count <= 2:
            return "MEDIUM"
        return "HIGH"

    def _relative_depth_for_bbox(self, depth_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.mean(roi))

    def _assign_person_ids(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active_ids: set[int] = set()
        assigned_detections: list[dict[str, Any]] = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            track_id = detection.get("track_id")
            matched_id = None
            best_distance = float("inf")

            if track_id is not None:
                matched_id = int(track_id)
            else:
                for previous_id, previous_center in self._track_positions.items():
                    distance = math.hypot(center_x - previous_center[0], center_y - previous_center[1])
                    if distance < best_distance:
                        best_distance = distance
                        matched_id = previous_id

            if matched_id is None or (track_id is None and best_distance > max(40.0, 0.15 * max(self.depth_estimator.model is not None, 0))):
                matched_id = self._next_person_id
                self._next_person_id += 1

            detection["person_id"] = matched_id
            active_ids.add(matched_id)
            self._track_positions[matched_id] = (center_x, center_y)
            assigned_detections.append(detection)

        stale_ids = set(self._track_positions) - active_ids
        for stale_id in stale_ids:
            del self._track_positions[stale_id]

        return assigned_detections

    def process(self, frame: np.ndarray):
        frame_start = time.perf_counter()

        depth_start = time.perf_counter()
        depth_map = self.depth_estimator.estimate(frame)
        depth_time = time.perf_counter() - depth_start

        detect_start = time.perf_counter()
        detections = self.detector.detect(frame)
        detection_time = time.perf_counter() - detect_start
        detections = self._assign_person_ids(detections)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            relative_depth = self._relative_depth_for_bbox(depth_map, (x1, y1, x2, y2))
            detection["relative_depth"] = relative_depth
            detection["depth_zone"] = self._depth_zone(relative_depth)

        self.last_detections = detections

        display_frame = frame.copy()
        nearest_person = None
        if detections:
            nearest_person = max(detections, key=lambda item: float(item["relative_depth"]))

        people_count = len(detections)
        occupancy = self._room_occupancy(people_count)

        cv2.putText(
            display_frame,
            "AI ROOM DEPTH & HUMAN MONITORING",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(display_frame, f"People: {people_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if nearest_person is not None:
            cv2.putText(
                display_frame,
                f"Nearest: Person {nearest_person['person_id']}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Nearest depth: {nearest_person['relative_depth']:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Zone: {nearest_person['depth_zone']}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        cv2.putText(display_frame, f"Room Occupancy: {occupancy}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Device: {self.device.upper()}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            person_id = detection["person_id"]
            relative_depth = float(detection["relative_depth"])
            depth_zone = detection["depth_zone"]
            confidence = float(detection["confidence"])

            box_color = (0, 255, 0)
            if depth_zone == "NEAR":
                box_color = (0, 0, 255)
            elif depth_zone == "MEDIUM":
                box_color = (0, 165, 255)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                display_frame,
                f"Person {person_id}",
                (x1, max(0, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

            if self.show_depth_info:
                cv2.putText(
                    display_frame,
                    f"{confidence:.2f} | {relative_depth:.2f} | {depth_zone}",
                    (x1, max(0, y1 - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

        if any(detection["depth_zone"] == "NEAR" for detection in detections):
            cv2.putText(
                display_frame,
                "WARNING: PERSON IN NEAR ZONE",
                (20, display_frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )

        depth_heatmap = create_depth_heatmap(depth_map, frame.shape) if self.show_heatmap else np.zeros_like(frame)
        if self.show_heatmap:
            blended = cv2.addWeighted(display_frame, 0.75, depth_heatmap, 0.25, 0)
        else:
            blended = display_frame.copy()

        elapsed = time.perf_counter() - frame_start
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            blended,
            "Relative depth only: monocular depth is not exact meters without calibration.",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(blended, f"FPS: {self.fps:.1f}", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(blended, f"Detection: {detection_time * 1000:.1f} ms", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(blended, f"Depth: {depth_time * 1000:.1f} ms", (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(blended, "Controls: Q quit | S screenshot | R record | H heatmap | D depth | C report", (20, blended.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if self.show_heatmap:
            side_by_side = np.hstack([blended, depth_heatmap])
        else:
            side_by_side = blended

        self.last_stats = {
            "people_count": people_count,
            "nearest_person": nearest_person,
            "occupancy": occupancy,
            "detection_time": detection_time,
            "depth_time": depth_time,
            "device": self.device,
        }
        return side_by_side, detections, depth_map
