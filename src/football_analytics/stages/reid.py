"""Player re-identification stage backed by SoccerNet sn-reid."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.contracts.schemas import (
    REID_EMBEDDINGS_SCHEMA,
    TRACK_REID_PROTOTYPES_SCHEMA,
    validate_mvp2_columns,
)
from football_analytics.integrations.sn_reid_extractor import SnReidExtractor
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    canonical_common,
    read_required_parquet,
)
from football_analytics.utils.io import write_json, write_rows_with_schema
from football_analytics.video.streaming import StreamingVideoReader


class ReidStage(Stage):
    name = "reid"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "tracks.parquet")
        read_required_parquet(self.run_dir / "track_quality.parquet")
        if not (self.run_dir / "input" / "test_clip.mp4").is_file():
            raise FileNotFoundError("working video missing")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        cfg = self.config.get("reid") or {}
        if not bool(cfg.get("enabled", True)):
            return self._write_empty(reason="reid.enabled=false")

        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        quality = read_required_parquet(self.run_dir / "track_quality.parquet")
        person_tracks = tracks[tracks["object_type"] == "person"].copy()
        if person_tracks.empty:
            return self._write_empty(reason="no person tracks")

        usable = set(
            quality.loc[quality["usable_for_metrics"].fillna(False), "track_id"]
            .astype(int)
            .tolist()
        )
        if usable:
            person_tracks = person_tracks[person_tracks["track_id"].isin(usable)]
        if person_tracks.empty:
            return self._write_empty(reason="no usable person tracks")

        sample_stride = max(1, int(cfg.get("sample_stride", 5)))
        max_samples = max(1, int(cfg.get("max_samples_per_track", 20)))
        min_area = float(
            cfg.get("min_bbox_area", self.config.get("geometry", {}).get("minimum_bbox_area", 36))
        )
        batch_size = max(1, int(cfg.get("batch_size", 32)))

        selected = self._select_samples(person_tracks, sample_stride, max_samples, min_area)
        if not selected:
            return self._write_empty(reason="no crops passed sampling gates")

        extractor = SnReidExtractor(
            model_name=str(cfg.get("model_name", "osnet_x1_0")),
            model_path=str(
                cfg.get(
                    "model_path",
                    "/home/ahmet/models/sn-reid/osnet_x1_0_market1501.pth",
                )
            ),
            sn_reid_root=str(
                cfg.get("sn_reid_root", "/home/ahmet/projects/soccernet/sn-reid")
            ),
            device=str(cfg.get("device", "cuda:0")),
            image_size=tuple(cfg.get("image_size", [256, 128])),
            verbose=bool(cfg.get("verbose", False)),
        )

        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in selected:
            by_frame[int(sample["frame_id"])].append(sample)

        embedding_rows: list[dict[str, Any]] = []
        pending_meta: list[dict[str, Any]] = []
        pending_crops: list[np.ndarray] = []
        reader = StreamingVideoReader(
            self.run_dir / "input" / "test_clip.mp4",
            chunk_seconds=float(self.config["runtime"]["chunk_seconds"]),
        )

        def flush() -> None:
            nonlocal pending_meta, pending_crops
            if not pending_crops:
                return
            vectors = extractor.extract(pending_crops, assume_bgr=True)
            for meta, vector in zip(pending_meta, vectors):
                embedding_rows.append(
                    {
                        **canonical_common(
                            self.run_dir,
                            int(meta["frame_id"]),
                            float(meta["timestamp_ms"]),
                            "sn_reid/osnet",
                            float(meta["confidence"]),
                            True,
                        ),
                        "track_id": int(meta["track_id"]),
                        "bbox_x1": float(meta["bbox_x1"]),
                        "bbox_y1": float(meta["bbox_y1"]),
                        "bbox_x2": float(meta["bbox_x2"]),
                        "bbox_y2": float(meta["bbox_y2"]),
                        "embedding": [float(x) for x in vector.tolist()],
                        "embedding_dim": int(vector.shape[0]),
                        "model_name": extractor.model_name,
                    }
                )
            pending_meta = []
            pending_crops = []

        for video_frame in reader:
            frame_id = int(video_frame.frame_id)
            samples = by_frame.get(frame_id)
            if not samples:
                continue
            image = video_frame.image
            height, width = image.shape[:2]
            for sample in samples:
                x1 = max(0, min(width - 1, int(round(sample["bbox_x1"]))))
                y1 = max(0, min(height - 1, int(round(sample["bbox_y1"]))))
                x2 = max(0, min(width, int(round(sample["bbox_x2"]))))
                y2 = max(0, min(height, int(round(sample["bbox_y2"]))))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pending_meta.append(sample)
                pending_crops.append(crop)
                if len(pending_crops) >= batch_size:
                    flush()
        flush()

        prototype_rows = self._build_prototypes(embedding_rows)

        embeddings_path = self.run_dir / "reid_embeddings.parquet"
        prototypes_path = self.run_dir / "track_reid_prototypes.parquet"
        write_rows_with_schema(embeddings_path, embedding_rows, REID_EMBEDDINGS_SCHEMA)
        write_rows_with_schema(
            prototypes_path, prototype_rows, TRACK_REID_PROTOTYPES_SCHEMA
        )
        metrics = {
            **extractor.info(),
            "selected_samples": len(selected),
            "written_embeddings": len(embedding_rows),
            "prototype_tracks": len(prototype_rows),
            "sample_stride": sample_stride,
            "max_samples_per_track": max_samples,
        }
        write_json(self.stage_dir / "metrics.json", metrics)
        return {
            "reid_embeddings": embeddings_path,
            "track_reid_prototypes": prototypes_path,
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        embeddings = pd.read_parquet(artifacts["reid_embeddings"])
        prototypes = pd.read_parquet(artifacts["track_reid_prototypes"])
        validate_mvp2_columns("reid_embeddings", list(embeddings.columns))
        validate_mvp2_columns("track_reid_prototypes", list(prototypes.columns))

    def _write_empty(self, *, reason: str) -> dict[str, Any]:
        embeddings_path = self.run_dir / "reid_embeddings.parquet"
        prototypes_path = self.run_dir / "track_reid_prototypes.parquet"
        write_rows_with_schema(embeddings_path, [], REID_EMBEDDINGS_SCHEMA)
        write_rows_with_schema(prototypes_path, [], TRACK_REID_PROTOTYPES_SCHEMA)
        write_json(
            self.stage_dir / "metrics.json",
            {"status": "empty", "reason": reason, "written_embeddings": 0},
        )
        return {
            "reid_embeddings": embeddings_path,
            "track_reid_prototypes": prototypes_path,
        }

    @staticmethod
    def _select_samples(
        tracks: pd.DataFrame,
        sample_stride: int,
        max_samples: int,
        min_area: float,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for track_id, group in tracks.groupby("track_id"):
            group = group.sort_values("frame_id")
            kept = 0
            for index, row in enumerate(group.itertuples(index=False)):
                if index % sample_stride != 0:
                    continue
                width = float(row.bbox_x2) - float(row.bbox_x1)
                height = float(row.bbox_y2) - float(row.bbox_y1)
                if width * height < min_area:
                    continue
                selected.append(
                    {
                        "frame_id": int(row.frame_id),
                        "timestamp_ms": float(row.timestamp_ms),
                        "track_id": int(track_id),
                        "bbox_x1": float(row.bbox_x1),
                        "bbox_y1": float(row.bbox_y1),
                        "bbox_x2": float(row.bbox_x2),
                        "bbox_y2": float(row.bbox_y2),
                        "confidence": float(row.tracking_confidence),
                    }
                )
                kept += 1
                if kept >= max_samples:
                    break
        return selected

    def _build_prototypes(
        self, embedding_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in embedding_rows:
            by_track[int(row["track_id"])].append(row)
        prototypes: list[dict[str, Any]] = []
        for track_id, rows in sorted(by_track.items()):
            matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            mean = matrix.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 1e-12:
                mean = mean / norm
            confidence = float(np.mean([row["confidence"] for row in rows]))
            frame_id = int(rows[len(rows) // 2]["frame_id"])
            timestamp_ms = float(rows[len(rows) // 2]["timestamp_ms"])
            prototypes.append(
                {
                    **canonical_common(
                        self.run_dir,
                        frame_id,
                        timestamp_ms,
                        "sn_reid/osnet_mean",
                        confidence,
                        True,
                    ),
                    "track_id": int(track_id),
                    "n_samples": int(len(rows)),
                    "embedding": [float(x) for x in mean.tolist()],
                    "embedding_dim": int(mean.shape[0]),
                    "model_name": str(rows[0]["model_name"]),
                }
            )
        return prototypes
