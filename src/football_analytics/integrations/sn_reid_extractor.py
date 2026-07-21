"""Thin wrapper around SoccerNet sn-reid ``FeatureExtractor``.

Imports torchreid from the locked local checkout so ``ai-dev`` keeps its own
CUDA torch stack. Embeddings are L2-normalized for cosine matching in the
global-identity layer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_SN_REID_ROOT = Path("/home/ahmet/projects/soccernet/sn-reid")
DEFAULT_MODEL_PATH = Path("/home/ahmet/models/sn-reid/osnet_x1_0_market1501.pth")


def _ensure_torchreid(sn_reid_root: Path) -> None:
    root = sn_reid_root.resolve()
    if not (root / "torchreid").is_dir():
        raise FileNotFoundError(f"sn-reid checkout missing torchreid/: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


class SnReidExtractor:
    """Crop → embedding adapter backed by sn-reid OSNet."""

    def __init__(
        self,
        *,
        model_name: str = "osnet_x1_0",
        model_path: str | Path = DEFAULT_MODEL_PATH,
        sn_reid_root: str | Path = DEFAULT_SN_REID_ROOT,
        device: str = "cuda",
        image_size: Sequence[int] = (256, 128),
        verbose: bool = False,
    ) -> None:
        root = Path(sn_reid_root)
        _ensure_torchreid(root)
        from torchreid.utils import FeatureExtractor  # noqa: WPS433

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ReID weights not found: {path}")

        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.model_name = model_name
        self.model_path = path
        self.device = device
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(path),
            image_size=self.image_size,
            device=device,
            verbose=verbose,
        )

    def extract(self, crops_bgr_or_rgb: Sequence[np.ndarray], *, assume_bgr: bool = True) -> np.ndarray:
        """Return L2-normalized float32 embeddings shaped ``(N, D)``."""
        if not crops_bgr_or_rgb:
            return np.zeros((0, 0), dtype=np.float32)
        images: list[np.ndarray] = []
        for crop in crops_bgr_or_rgb:
            if crop.ndim != 3 or crop.shape[2] != 3:
                raise ValueError(f"expected HxWx3 crop, got {getattr(crop, 'shape', None)}")
            rgb = crop[:, :, ::-1].copy() if assume_bgr else np.ascontiguousarray(crop)
            images.append(rgb)
        features = self._extractor(images)
        array = features.detach().float().cpu().numpy().astype(np.float32, copy=False)
        return l2_normalize(array).astype(np.float32, copy=False)

    def info(self) -> dict[str, Any]:
        return {
            "backend": "sn_reid",
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "device": self.device,
            "image_size": list(self.image_size),
        }
