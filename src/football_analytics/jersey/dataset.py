"""SoccerNet jersey tracklet discovery, splitting, and loading."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from PIL import Image, ImageStat
from torch.utils.data import Dataset

from .schemas import TrackletRecord, label_to_class
from .transforms import build_transform

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _stable_score(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_ground_truth(dataset_root: str | Path, split: str) -> dict[str, int]:
    root = Path(dataset_root)
    path = root / split / f"{split}_gt.json"
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    labels = {str(key): int(value) for key, value in raw.items()}
    for label in labels.values():
        label_to_class(label)
    return labels


def deterministic_tracklet_split(
    tracklet_ids: Iterable[str], *, val_fraction: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split whole tracklets reproducibly; no frame can occur in both outputs."""
    ids = sorted(set(map(str, tracklet_ids)))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(ids) < 2:
        return ids, []
    ordered = sorted(ids, key=lambda value: (_stable_score(value, seed), value))
    val_count = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
    val = sorted(ordered[:val_count])
    train = sorted(ordered[val_count:])
    assert set(train).isdisjoint(val)
    return train, val


def select_tracklets(
    tracklet_ids: Iterable[str], subset_size: int | None, *, seed: int = 42
) -> list[str]:
    ids = sorted(set(map(str, tracklet_ids)))
    if subset_size is None or subset_size <= 0 or subset_size >= len(ids):
        return ids
    return sorted(ids, key=lambda value: (_stable_score(value, seed), value))[:subset_size]


def _frame_sort_key(path: Path) -> tuple[int, str]:
    tail = path.stem.rsplit("_", 1)[-1]
    return (int(tail) if tail.isdigit() else 0, path.name)


def build_records(
    dataset_root: str | Path,
    split: str,
    tracklet_ids: Sequence[str] | None = None,
    *,
    subset_size: int | None = None,
    seed: int = 42,
    include_unknown: bool = True,
) -> list[TrackletRecord]:
    root = Path(dataset_root)
    labels = load_ground_truth(root, split)
    ids = list(tracklet_ids) if tracklet_ids is not None else list(labels)
    if not include_unknown:
        ids = [tracklet_id for tracklet_id in ids if labels[tracklet_id] != -1]
    ids = select_tracklets(ids, subset_size, seed=seed)
    records: list[TrackletRecord] = []
    for tracklet_id in ids:
        folder = root / split / "images" / tracklet_id
        if tracklet_id not in labels:
            raise KeyError(f"tracklet {tracklet_id!r} is absent from {split}_gt.json")
        frames = tuple(
            sorted(
                (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
                key=_frame_sort_key,
            )
        )
        if frames:
            records.append(TrackletRecord(tracklet_id, labels[tracklet_id], frames, split))
    return records


def sample_frame_paths(paths: Sequence[Path], num_frames: int) -> tuple[Path, ...]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not paths:
        return ()
    if len(paths) <= num_frames:
        indices = [round(index * (len(paths) - 1) / max(num_frames - 1, 1))
                   for index in range(num_frames)]
    else:
        indices = [round((index + 0.5) * len(paths) / num_frames - 0.5)
                   for index in range(num_frames)]
    return tuple(paths[min(len(paths) - 1, max(0, index))] for index in indices)


def estimate_frame_quality(image: Image.Image, target_area: int) -> float:
    gray = image.convert("L")
    contrast = min(1.0, float(ImageStat.Stat(gray).stddev[0]) / 48.0)
    resolution = min(1.0, (image.width * image.height / max(target_area, 1)) ** 0.5)
    return max(0.05, 0.55 * resolution + 0.45 * contrast)


class JerseyTrackletDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: Sequence[TrackletRecord],
        *,
        num_frames: int = 8,
        image_size: tuple[int, int] = (128, 64),
        training: bool = False,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.records = list(records)
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform or build_transform(image_size, training=training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        selected = sample_frame_paths(record.frame_paths, self.num_frames)
        frames: list[torch.Tensor] = []
        qualities: list[float] = []
        valid: list[bool] = []
        fallback = torch.zeros(3, *self.image_size, dtype=torch.float32)
        for path in selected:
            try:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                qualities.append(estimate_frame_quality(image, self.image_size[0] * self.image_size[1]))
                frames.append(self.transform(image))
                valid.append(True)
            except (OSError, ValueError):
                frames.append(fallback.clone())
                qualities.append(0.0)
                valid.append(False)
        if not frames:
            frames, qualities, valid = [fallback], [0.0], [False]
        return {
            "frames": torch.stack(frames),
            "frame_quality": torch.tensor(qualities, dtype=torch.float32),
            "frame_mask": torch.tensor(valid, dtype=torch.bool),
            "label": torch.tensor(label_to_class(record.label), dtype=torch.long),
            "tracklet_id": record.tracklet_id,
            "frame_paths": tuple(map(str, selected)),
        }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
