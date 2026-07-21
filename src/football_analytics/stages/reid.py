"""Player re-identification stage backed by SoccerNet sn-reid."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.reid_matching import (
    crop_quality_score,
    robust_mean_embedding,
    torso_crop_xyxy,
)
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

        person_tracks = self._filter_tracks_for_reid(person_tracks, quality, cfg)
        if person_tracks.empty:
            return self._write_empty(reason="no tracks passed reid coverage gates")

        sample_stride = max(1, int(cfg.get("sample_stride", 4)))
        max_samples = max(1, int(cfg.get("max_samples_per_track", 24)))
        min_area = float(
            cfg.get(
                "min_bbox_area",
                self.config.get("geometry", {}).get("minimum_bbox_area", 36),
            )
        )
        batch_size = max(1, int(cfg.get("batch_size", 32)))
        use_torso = bool(cfg.get("use_torso_crop", True))
        image_width = int(cfg.get("assume_width", 1920))
        image_height = int(cfg.get("assume_height", 1080))

        selected = self._select_samples(
            person_tracks,
            sample_stride=sample_stride,
            max_samples=max_samples,
            min_area=min_area,
            image_width=image_width,
            image_height=image_height,
        )
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
                if use_torso:
                    crop_box = torso_crop_xyxy(
                        (
                            sample["bbox_x1"],
                            sample["bbox_y1"],
                            sample["bbox_x2"],
                            sample["bbox_y2"],
                        ),
                        image_width=width,
                        image_height=height,
                    )
                    if crop_box is None:
                        continue
                    x1, y1, x2, y2 = crop_box
                else:
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

        prototype_rows, consistency_by_track = self._build_prototypes(embedding_rows)

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
            "use_torso_crop": use_torso,
            "coverage_mode": str(cfg.get("coverage_mode", "broad")),
            "mean_prototype_consistency": float(np.mean(list(consistency_by_track.values())))
            if consistency_by_track
            else 0.0,
        }
        write_json(
            self.stage_dir / "prototype_consistency.json",
            {
                "tracks": [
                    {"track_id": tid, "consistency": cons}
                    for tid, cons in sorted(consistency_by_track.items())
                ]
            },
        )
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
    def _filter_tracks_for_reid(
        person_tracks: pd.DataFrame,
        quality: pd.DataFrame,
        cfg: dict[str, Any],
    ) -> pd.DataFrame:
        """Broad coverage beyond usable_for_metrics-only tracks."""
        coverage_mode = str(cfg.get("coverage_mode", "broad")).lower()
        min_frames = max(1, int(cfg.get("min_track_frames", 5)))
        quality_by: dict[int, Any] = {}
        if not quality.empty and "track_id" in quality.columns:
            for row in quality.itertuples(index=False):
                quality_by[int(row.track_id)] = row

        keep_ids: set[int] = set()
        lengths = person_tracks.groupby("track_id").size().to_dict()
        for track_id, length in lengths.items():
            tid = int(track_id)
            qrow = quality_by.get(tid)
            usable = bool(getattr(qrow, "usable_for_metrics", False)) if qrow is not None else False
            if coverage_mode == "strict_usable":
                if usable:
                    keep_ids.add(tid)
                continue
            if usable or int(length) >= min_frames:
                if qrow is not None:
                    truncation = float(getattr(qrow, "border_truncation", 0.0) or 0.0)
                    if truncation > float(cfg.get("max_border_truncation", 0.85)):
                        continue
                keep_ids.add(tid)

        if not keep_ids:
            return person_tracks.iloc[0:0]
        return person_tracks[person_tracks["track_id"].isin(keep_ids)].copy()

    @staticmethod
    def _select_samples(
        tracks: pd.DataFrame,
        *,
        sample_stride: int,
        max_samples: int,
        min_area: float,
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for track_id, group in tracks.groupby("track_id"):
            group = group.sort_values("frame_id")
            candidates: list[dict[str, Any]] = []
            for index, row in enumerate(group.itertuples(index=False)):
                if index % sample_stride != 0:
                    continue
                width = float(row.bbox_x2) - float(row.bbox_x1)
                height = float(row.bbox_y2) - float(row.bbox_y1)
                if width * height < min_area:
                    continue
                quality = crop_quality_score(
                    (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
                    image_width=image_width,
                    image_height=image_height,
                    tracking_confidence=float(row.tracking_confidence),
                )
                candidates.append(
                    {
                        "frame_id": int(row.frame_id),
                        "timestamp_ms": float(row.timestamp_ms),
                        "track_id": int(track_id),
                        "bbox_x1": float(row.bbox_x1),
                        "bbox_y1": float(row.bbox_y1),
                        "bbox_x2": float(row.bbox_x2),
                        "bbox_y2": float(row.bbox_y2),
                        "confidence": float(row.tracking_confidence),
                        "quality": quality,
                    }
                )
            if not candidates:
                continue
            candidates.sort(key=lambda item: item["quality"], reverse=True)
            picked = candidates[:max_samples]
            picked.sort(key=lambda item: item["frame_id"])
            selected.extend(picked)
        return selected

    def _build_prototypes(
        self, embedding_rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[int, float]]:
        by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in embedding_rows:
            by_track[int(row["track_id"])].append(row)
        prototypes: list[dict[str, Any]] = []
        consistency_by_track: dict[int, float] = {}
        for track_id, rows in sorted(by_track.items()):
            vectors = [np.asarray(row["embedding"], dtype=np.float32) for row in rows]
            mean, consistency = robust_mean_embedding(vectors)
            if mean is None:
                continue
            confidence = float(np.mean([row["confidence"] for row in rows]))
            confidence = float(confidence * (0.5 + 0.5 * consistency))
            frame_id = int(rows[len(rows) // 2]["frame_id"])
            timestamp_ms = float(rows[len(rows) // 2]["timestamp_ms"])
            prototypes.append(
                {
                    **canonical_common(
                        self.run_dir,
                        frame_id,
                        timestamp_ms,
                        "sn_reid/osnet_robust_median",
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
            consistency_by_track[int(track_id)] = float(consistency)
        return prototypes, consistency_by_track
