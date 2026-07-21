"""Team identity stage — SigLIP (Roboflow sports) with kit-family fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_analytics.analytics.kit_descriptor import bbox_contamination
from football_analytics.analytics.role_identity import PlayerRole
from football_analytics.analytics.team_identity import TeamIdentityAssigner, sample_upper_torso
from football_analytics.analytics.team_lock import lock_track_teams
from football_analytics.contracts.schemas import (
    TRACK_IDENTITIES_SCHEMA,
    validate_mvp2_columns,
)
from football_analytics.geometry.bbox import BBox
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import (
    canonical_common,
    read_required_parquet,
    video_fps,
)
from football_analytics.utils.io import write_rows_with_schema
from football_analytics.video.streaming import StreamingVideoReader


class TeamIdentityStage(Stage):
    name = "team_identity"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "tracks.parquet")
        read_required_parquet(self.run_dir / "track_quality.parquet")
        read_required_parquet(self.run_dir / "shot_segments.parquet")
        if not (self.run_dir / "input" / "test_clip.mp4").is_file():
            raise FileNotFoundError("working video missing")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        cfg = self.config["team_identity"]
        method = str(cfg.get("method", "siglip")).lower()
        video = self.run_dir / "input" / "test_clip.mp4"
        fps = video_fps(self.run_dir)

        if method in {"siglip", "sports", "auto"}:
            try:
                rows = self._run_siglip(tracks, video, cfg, fps)
                source = "siglip_umap_kmeans"
            except Exception as exc:  # noqa: BLE001 — fall back cleanly
                if method != "auto" and method != "siglip":
                    raise
                # auto/siglip: fall back to kit families if model unavailable
                rows = self._run_kit_family(tracks, video, cfg, fps)
                source = f"kit_family_fallback:{type(exc).__name__}"
        else:
            rows = self._run_kit_family(tracks, video, cfg, fps)
            source = "kit_family_torso_kmeans"

        # Rewrite source_method for auditability
        for row in rows:
            if source.startswith("kit_family_fallback"):
                row["source_method"] = source
            elif source == "siglip_umap_kmeans":
                row["source_method"] = source

        output = self.run_dir / "track_identities.parquet"
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = lock_track_teams(
                frame, min_confidence=float(cfg.get("minimum_team_confidence", 0.35))
            )
            # Spatial centroid resolve for still-unknown (sports-main GK trick).
            frame = self._resolve_unknowns_by_centroid(frame, tracks)
            rows = frame.to_dict("records")
        write_rows_with_schema(output, rows, TRACK_IDENTITIES_SCHEMA)
        return {"track_identities": output, "team_method": source}

    def _run_siglip(
        self,
        tracks: pd.DataFrame,
        video,
        cfg: dict[str, Any],
        fps: float,
    ) -> list[dict[str, Any]]:
        from football_analytics.analytics.team_classifier_siglip import (
            assign_teams_with_classifier,
            fit_team_classifier_from_video,
        )

        device = str(cfg.get("siglip_device", "cuda"))
        clf = fit_team_classifier_from_video(
            str(video),
            tracks,
            stride=int(cfg.get("siglip_fit_stride", 30)),
            max_crops=int(cfg.get("siglip_max_crops", 400)),
            device=device,
            batch_size=int(cfg.get("siglip_batch_size", 32)),
            exclude_dark=bool(cfg.get("exclude_dark_kits", True)),
        )
        exclude_dark = bool(cfg.get("exclude_dark_kits", True))
        assignments, dark_tracks = assign_teams_with_classifier(
            str(video),
            tracks,
            clf,
            predict_stride=int(cfg.get("siglip_predict_stride", 5)),
            min_votes=int(cfg.get("siglip_min_votes", 2)),
            exclude_dark=exclude_dark,
        )
        assignments = self._stabilize_team_labels(assignments, tracks, video)

        person = tracks[tracks["object_type"].eq("person")]
        rows: list[dict[str, Any]] = []
        for row in person.itertuples(index=False):
            tid = int(row.track_id)
            frame_id = int(row.frame_id)
            if tid in dark_tracks:
                team_id = None
                conf = 0.85
                role = "referee"
            else:
                team_tuple = assignments.get(tid)
                if team_tuple is None:
                    team_id = None
                    conf = 0.0
                    role = "unknown"
                else:
                    team_idx, conf = team_tuple
                    team_id = f"team_{int(team_idx)}"
                    role = "unknown"
                    if conf < float(cfg.get("minimum_team_confidence", 0.35)):
                        team_id = None
            rows.append(
                {
                    **canonical_common(
                        self.run_dir,
                        frame_id,
                        frame_id * 1000.0 / fps,
                        "siglip_umap_kmeans",
                        float(conf),
                        team_id is not None,
                    ),
                    "track_id": tid,
                    "role": role,
                    "role_confidence": float(conf) if role == "referee" else 0.0,
                    "team_id": team_id,
                    "team_confidence": float(conf) if team_id is not None else 0.0,
                    "color_quality": float(conf),
                    "temporal_consistency": float(conf),
                }
            )
        return rows

    @staticmethod
    def _stabilize_team_labels(
        assignments: dict[int, tuple[int, float]],
        tracks: pd.DataFrame,
        video,
    ) -> dict[int, tuple[int, float]]:
        """Flip SigLIP 0/1 so team_0 is the lighter / whiter kit on average."""
        import cv2
        from football_analytics.analytics.kit_descriptor import (
            kit_feature_from_frame,
            white_score,
            colored_score,
        )

        if not assignments:
            return assignments
        person = tracks[tracks["object_type"].eq("person")]
        # Sample one mid frame crop per assigned track
        samples: dict[int, list[float]] = {0: [], 1: []}
        by_tid = {int(tid): g for tid, g in person.groupby("track_id")}
        cap = cv2.VideoCapture(str(video))
        for tid, (team, _conf) in list(assignments.items())[:80]:
            g = by_tid.get(int(tid))
            if g is None or g.empty:
                continue
            mid = g.iloc[len(g) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid.frame_id))
            ok, frame = cap.read()
            if not ok:
                continue
            feat, _ = kit_feature_from_frame(
                frame, (mid.bbox_x1, mid.bbox_y1, mid.bbox_x2, mid.bbox_y2)
            )
            if feat is None:
                continue
            samples[int(team)].append(white_score(feat) - colored_score(feat))
        cap.release()
        if not samples[0] or not samples[1]:
            return assignments
        score0 = float(np.mean(samples[0]))
        score1 = float(np.mean(samples[1]))
        # Higher white-minus-color should be team_0
        if score1 > score0:
            return {tid: (1 - team, conf) for tid, (team, conf) in assignments.items()}
        return assignments

    def _run_kit_family(
        self,
        tracks: pd.DataFrame,
        video,
        cfg: dict[str, Any],
        fps: float,
    ) -> list[dict[str, Any]]:
        top_ratio = float(cfg.get("upper_body_top_ratio", 0.15))
        bottom_ratio = float(cfg.get("upper_body_bottom_ratio", 0.65))
        side_inset = float(cfg.get("side_inset", 0.20))
        max_contamination = float(cfg.get("max_bbox_contamination", 0.35))
        min_fraction = float(
            cfg.get(
                "minimum_color_pixel_fraction",
                min(0.08, float(self.config["geometry"]["minimum_crop_quality"])),
            )
        )
        assigner = TeamIdentityAssigner(
            min_samples=int(cfg["minimum_team_training_samples"]),
            min_tracks=max(2, min(4, int(cfg["minimum_team_training_samples"]) // 2)),
            history_size=int(cfg["maximum_samples_per_track"]),
            min_valid_pixel_fraction=min_fraction,
            unknown_confidence_threshold=float(cfg["minimum_team_confidence"]),
            exclude_dark_kits=bool(cfg.get("exclude_dark_kits", True)),
        )
        grouped = {int(key): value for key, value in tracks.groupby("frame_id")}
        rows: list[dict[str, Any]] = []
        reader = StreamingVideoReader(
            video, chunk_seconds=float(self.config["runtime"]["chunk_seconds"])
        )
        for video_frame in reader:
            frame_id = video_frame.frame_id
            image = video_frame.image
            frame_tracks = grouped.get(frame_id)
            if frame_tracks is None:
                continue
            frame_boxes = [
                (
                    int(item.track_id),
                    (
                        float(item.bbox_x1),
                        float(item.bbox_y1),
                        float(item.bbox_x2),
                        float(item.bbox_y2),
                    ),
                )
                for item in frame_tracks.itertuples()
            ]
            observations: list[tuple[Any, float]] = []
            for item in frame_tracks.itertuples():
                box = BBox(
                    float(item.bbox_x1),
                    float(item.bbox_y1),
                    float(item.bbox_x2),
                    float(item.bbox_y2),
                )
                target = (
                    float(item.bbox_x1),
                    float(item.bbox_y1),
                    float(item.bbox_x2),
                    float(item.bbox_y2),
                )
                others = [coords for tid, coords in frame_boxes if tid != int(item.track_id)]
                contaminated = bbox_contamination(target, others) > max_contamination
                color_quality = 0.0
                if not contaminated:
                    feature, color_quality = sample_upper_torso(
                        image,
                        box,
                        top_ratio=top_ratio,
                        bottom_ratio=bottom_ratio,
                        side_inset=side_inset,
                    )
                    if feature is not None:
                        assigner.add_sample(
                            int(item.track_id),
                            feature,
                            role=PlayerRole.UNKNOWN,
                            valid_pixel_fraction=color_quality,
                        )
                observations.append((item, color_quality))
            assigner.fit()
            for item, color_quality in observations:
                assignment = assigner.assign(int(item.track_id), role=PlayerRole.UNKNOWN)
                team_id = (
                    f"team_{assignment.team_id}"
                    if assignment.team_id is not None
                    else None
                )
                role_name = (
                    "referee" if assignment.role is PlayerRole.REFEREE else "unknown"
                )
                rows.append(
                    {
                        **canonical_common(
                            self.run_dir,
                            int(item.frame_id),
                            int(item.frame_id) * 1000.0 / fps,
                            "kit_family_torso_kmeans",
                            assignment.confidence,
                            team_id is not None,
                        ),
                        "track_id": int(item.track_id),
                        "role": role_name,
                        "role_confidence": 0.85 if role_name == "referee" else 0.0,
                        "team_id": team_id,
                        "team_confidence": float(assignment.confidence),
                        "color_quality": float(color_quality),
                        "temporal_consistency": float(assignment.temporal_consistency),
                    }
                )
        return rows

    @staticmethod
    def _resolve_unknowns_by_centroid(
        identities: pd.DataFrame, tracks: pd.DataFrame
    ) -> pd.DataFrame:
        """sports-main style: assign unknowns to nearest team centroid per frame."""
        if identities.empty:
            return identities
        out = identities.copy()
        person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
        boxes = person.set_index(["frame_id", "track_id"])[
            ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        ]

        for frame_id, group in out.groupby("frame_id"):
            known = group[group["team_id"].notna()]
            unknown = group[group["team_id"].isna()]
            if "role" in unknown.columns:
                unknown = unknown[unknown["role"].astype(str) != "referee"]
            if known.empty or unknown.empty:
                continue
            known_xy = []
            known_team = []
            for r in known.itertuples(index=False):
                key = (int(frame_id), int(r.track_id))
                if key not in boxes.index:
                    continue
                b = boxes.loc[key]
                if isinstance(b, pd.DataFrame):
                    b = b.iloc[0]
                known_xy.append(
                    [
                        0.5 * (float(b.bbox_x1) + float(b.bbox_x2)),
                        float(b.bbox_y2),
                    ]
                )
                known_team.append(0 if str(r.team_id).endswith("0") else 1)
            if len(known_xy) < 2 or len(set(known_team)) < 2:
                continue
            known_xy_a = np.asarray(known_xy, dtype=np.float64)
            known_team_a = np.asarray(known_team, dtype=np.int32)
            c0 = known_xy_a[known_team_a == 0].mean(axis=0)
            c1 = known_xy_a[known_team_a == 1].mean(axis=0)
            for r in unknown.itertuples(index=False):
                key = (int(frame_id), int(r.track_id))
                if key not in boxes.index:
                    continue
                b = boxes.loc[key]
                if isinstance(b, pd.DataFrame):
                    b = b.iloc[0]
                xy = np.asarray(
                    [
                        0.5 * (float(b.bbox_x1) + float(b.bbox_x2)),
                        float(b.bbox_y2),
                    ],
                    dtype=np.float64,
                )
                team = 0 if np.linalg.norm(xy - c0) <= np.linalg.norm(xy - c1) else 1
                mask = (out["frame_id"] == frame_id) & (out["track_id"] == int(r.track_id))
                out.loc[mask, "team_id"] = f"team_{team}"
                out.loc[mask, "team_confidence"] = 0.45
                out.loc[mask, "valid"] = True
                out.loc[mask, "role"] = "unknown"
        return out

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        frame = pd.read_parquet(artifacts["track_identities"])
        validate_mvp2_columns("track_identities", list(frame.columns))
