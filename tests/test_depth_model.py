from __future__ import annotations

import numpy as np

from src.depth_model import MonoDepthEstimator


def test_depth_model_loads_and_returns_relative_depth():
    estimator = MonoDepthEstimator(device="cpu")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    depth_map = estimator.estimate(frame)

    assert depth_map.shape == (224, 224)
    assert 0.0 <= float(depth_map.min()) <= 1.0
    assert 0.0 <= float(depth_map.max()) <= 1.0
    assert not np.isnan(depth_map).any()
