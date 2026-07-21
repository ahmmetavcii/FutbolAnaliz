"""Checkpoint loading and confidence-aware jersey inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .dataset import IMAGE_SUFFIXES, JerseyTrackletDataset
from .model import TemporalJerseyRecognizer, build_model
from .schemas import (
    UNKNOWN_CLASS_INDEX,
    UNKNOWN_LABEL,
    JerseyPrediction,
    TrackletRecord,
    class_to_label,
)
from .train import resolve_device


def load_model_checkpoint(
    checkpoint_path: str | Path, device: str | torch.device = "auto"
) -> tuple[TemporalJerseyRecognizer, dict[str, Any], torch.device]:
    resolved = resolve_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = build_model(config.get("model", {}))
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.to(resolved).eval()
    return model, config, resolved


def record_from_directory(path: str | Path, tracklet_id: str | None = None) -> TrackletRecord:
    folder = Path(path)
    frames = tuple(
        sorted(item for item in folder.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    )
    if not frames:
        raise ValueError(f"no images found in {folder}")
    return TrackletRecord(tracklet_id or folder.name, UNKNOWN_LABEL, frames, "inference")


@torch.inference_mode()
def predict_records(
    model: TemporalJerseyRecognizer,
    records: Sequence[TrackletRecord],
    *,
    device: torch.device,
    num_frames: int = 8,
    image_size: tuple[int, int] = (128, 64),
    batch_size: int = 16,
    confidence_threshold: float = 0.55,
    workers: int = 0,
) -> list[JerseyPrediction]:
    dataset = JerseyTrackletDataset(
        records, num_frames=num_frames, image_size=image_size, training=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    predictions: list[JerseyPrediction] = []
    for batch in loader:
        frames = batch["frames"].to(device)
        mask = batch["frame_mask"].to(device)
        quality = batch["frame_quality"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(frames, mask, quality)
            probabilities = output["logits"].softmax(dim=1)
        raw_confidence, classes = probabilities.max(dim=1)
        for index, tracklet_id in enumerate(batch["tracklet_id"]):
            predicted_class = int(classes[index])
            raw = float(raw_confidence[index])
            label = class_to_label(predicted_class)
            if predicted_class == UNKNOWN_CLASS_INDEX or raw < confidence_threshold:
                label = UNKNOWN_LABEL
            valid_frames = int(mask[index].sum())
            weights = output["frame_weights"][index].detach().float().cpu().tolist()
            predictions.append(
                JerseyPrediction(
                    tracklet_id=str(tracklet_id),
                    jersey_number=label,
                    confidence=raw,
                    predicted_class=predicted_class,
                    raw_confidence=raw,
                    num_frames=valid_frames,
                    frame_weights=tuple(float(value) for value in weights),
                )
            )
    return predictions


def run_inference(
    checkpoint_path: str | Path,
    records: Sequence[TrackletRecord],
    *,
    device: str = "auto",
    confidence_threshold: float | None = None,
    output_path: str | Path | None = None,
) -> list[JerseyPrediction]:
    model, config, resolved = load_model_checkpoint(checkpoint_path, device)
    data_cfg = config.get("dataset", {})
    inference_cfg = config.get("inference", {})
    predictions = predict_records(
        model,
        records,
        device=resolved,
        num_frames=int(data_cfg.get("num_frames", 8)),
        image_size=tuple(map(int, data_cfg.get("image_size", [128, 64]))),
        batch_size=int(inference_cfg.get("batch_size", 16)),
        confidence_threshold=float(
            confidence_threshold
            if confidence_threshold is not None
            else inference_cfg.get("confidence_threshold", 0.55)
        ),
        workers=int(config.get("runtime", {}).get("workers", 0)),
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([prediction.to_dict() for prediction in predictions], indent=2),
            encoding="utf-8",
        )
    return predictions
