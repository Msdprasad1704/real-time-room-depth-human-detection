from __future__ import annotations

import cv2
import numpy as np


def create_depth_heatmap(depth_map: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
    """Maps normalized relative depth values to a colorized heatmap."""
    depth_map = np.asarray(depth_map, dtype=np.float32)
    normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    grayscale = np.uint8(normalized)
    heatmap = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)

    if heatmap.shape[:2] != frame_shape[:2]:
        heatmap = cv2.resize(heatmap, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_LINEAR)

    return heatmap
