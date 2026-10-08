from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import cv2
import numpy as np
import torch


MISSING_CHECKPOINT_MESSAGE = (
    "Depth model checkpoint is missing. Please download the pretrained checkpoint "
    "and place it in models/."
)
DOWNLOAD_INSTRUCTIONS = (
    "Download the MiDaS small checkpoint from: "
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt "
    "and save it as models/midas_small.pt"
)


class MonoDepthEstimator:
    """Loads a pretrained monocular depth model and estimates relative depth."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.transform: Callable[[np.ndarray], torch.Tensor] | None = None
        self.checkpoint_path = Path(__file__).resolve().parents[1] / "models" / "midas_small.pt"
        self._configure_torch_hub_cache()
        self._load_model()

    def _configure_torch_hub_cache(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        hub_cache_dir = project_root / "models" / "torch_hub"
        hub_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_cache_dir))

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"{MISSING_CHECKPOINT_MESSAGE} {DOWNLOAD_INSTRUCTIONS}")

        repo_name = "intel-isl/MiDaS"
        try:
            loaded_model = cast(Any, torch.hub.load(repo_name, "MiDaS_small", pretrained=False, trust_repo="check"))
            loaded_transforms = cast(Any, torch.hub.load(repo_name, "transforms", trust_repo="check"))
            self.model = cast(torch.nn.Module, loaded_model)
            self.transform = cast(Callable[[np.ndarray], torch.Tensor], loaded_transforms.small_transform)

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()

            if self.model is not None and hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the MiDaS depth-estimation model from the local checkpoint. "
                f"Original error: {type(exc).__name__}: {exc}. {DOWNLOAD_INSTRUCTIONS}"
            ) from exc

        if self.model is None:
            raise RuntimeError("MiDaS model could not be initialized.")

        self.model.eval()
        self.model.to(self.device)

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        if self.transform is None or self.model is None:
            raise RuntimeError("Depth model is not initialized.")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb_image).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(0)

        prediction = prediction.squeeze().cpu().numpy()
        if prediction.ndim == 0:
            prediction = np.asarray([float(prediction)], dtype=np.float32)
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]

        depth_map = np.asarray(prediction, dtype=np.float32)
        if depth_map.shape != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))
        if np.isclose(depth_max, depth_min):
            return np.zeros_like(depth_map, dtype=np.float32)

        depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        return depth_map
