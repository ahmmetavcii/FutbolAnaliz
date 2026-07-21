from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from football_analytics.jersey.dataset import (
    JerseyTrackletDataset,
    build_records,
    deterministic_tracklet_split,
)
from football_analytics.jersey.evaluate import evaluate_predictions
from football_analytics.jersey.infer import predict_records
from football_analytics.jersey.model import TemporalJerseyRecognizer
from football_analytics.jersey.schemas import JerseyPrediction, UNKNOWN_CLASS_INDEX
from football_analytics.jersey.temporal_pooling import QualityWeightedTemporalPooling


def _make_dataset(root: Path) -> list:
    split = root / "train"
    (split / "images").mkdir(parents=True)
    labels = {"track-a": 7, "track-b": -1, "track-c": 42, "track-d": 0}
    (split / "train_gt.json").write_text(json.dumps(labels), encoding="utf-8")
    for index, tracklet_id in enumerate(labels):
        folder = split / "images" / tracklet_id
        folder.mkdir()
        for frame in range(3):
            Image.new("RGB", (24 + index, 48), (30 * index, 40, 80)).save(
                folder / f"{tracklet_id}_{frame}.jpg"
            )
    return build_records(root, "train")


def test_tracklet_split_is_deterministic_and_has_no_leakage(tmp_path: Path) -> None:
    records = _make_dataset(tmp_path)
    ids = [record.tracklet_id for record in records]
    first = deterministic_tracklet_split(ids, val_fraction=0.25, seed=9)
    second = deterministic_tracklet_split(reversed(ids), val_fraction=0.25, seed=9)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(ids)
    sample = JerseyTrackletDataset(records, num_frames=2, image_size=(64, 32))[1]
    assert sample["frames"].shape == (2, 3, 64, 32)
    assert int(sample["label"]) == UNKNOWN_CLASS_INDEX


def test_quality_pooling_and_compact_model_forward() -> None:
    pooling = QualityWeightedTemporalPooling(4, hidden_dim=2)
    for parameter in pooling.parameters():
        torch.nn.init.zeros_(parameter)
    features = torch.tensor([[[1.0] * 4, [3.0] * 4]])
    pooled, weights = pooling(features, frame_quality=torch.tensor([[0.1, 0.9]]))
    assert weights[0, 1] > weights[0, 0]
    assert torch.allclose(pooled, torch.full((1, 4), 2.8), atol=1e-4)

    model = TemporalJerseyRecognizer(width_mult=0.25).eval()
    with torch.inference_mode():
        output = model(
            torch.randn(2, 3, 3, 64, 32),
            torch.tensor([[True, True, True], [True, False, True]]),
            torch.ones(2, 3),
        )
    assert output["logits"].shape == (2, 101)
    assert torch.allclose(output["frame_weights"].sum(1), torch.ones(2), atol=1e-5)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


class _FixedModel(torch.nn.Module):
    def forward(self, frames, frame_mask=None, frame_quality=None):
        logits = torch.zeros((frames.shape[0], 101), device=frames.device)
        logits[:, 12] = 1.0
        return {
            "logits": logits,
            "frame_weights": frame_mask.float() / frame_mask.float().sum(1, keepdim=True),
        }


def test_inference_rejects_low_confidence_and_evaluation_counts_unknown(tmp_path: Path) -> None:
    records = _make_dataset(tmp_path)
    predictions = predict_records(
        _FixedModel(),
        records[:2],
        device=torch.device("cpu"),
        num_frames=2,
        image_size=(64, 32),
        confidence_threshold=0.5,
    )
    assert [prediction.jersey_number for prediction in predictions] == [-1, -1]
    metrics, matrix = evaluate_predictions([7, -1], predictions)
    assert matrix.shape == (101, 101)
    assert metrics["samples"] == 2
    assert metrics["unknown_recall"] == 1.0

    high_confidence = JerseyPrediction("x", 7, 0.9, 7, 0.9, 2)
    metrics, _ = evaluate_predictions([7], [high_confidence])
    assert np.isclose(metrics["accuracy"], 1.0)
