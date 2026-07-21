"""SigLIP + UMAP + KMeans team classifier (ported from Roboflow sports).

Reference: https://github.com/roboflow/sports ``sports/common/team.py``

Fits once on sampled player crops, then predicts team labels. Goalkeepers /
dark outliers can be resolved by proximity to team centroids.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

import cv2
import numpy as np

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(sequence: Sequence, batch_size: int):
    batch_size = max(int(batch_size), 1)
    batch: list = []
    for element in sequence:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(element)
    if batch:
        yield batch


def crop_person_bgr(
    frame_bgr: np.ndarray,
    bbox: Sequence[float],
    *,
    pad: float = 0.05,
) -> np.ndarray | None:
    """Full-body crop (sports-main style), lightly padded."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def resolve_by_team_centroids(
    player_xy: np.ndarray,
    player_team_id: np.ndarray,
    query_xy: np.ndarray,
) -> np.ndarray:
    """Assign queries to nearest team centroid (sports-main GK resolver)."""
    if len(query_xy) == 0:
        return np.array([], dtype=np.int32)
    team_ids = np.unique(player_team_id)
    if len(team_ids) == 0:
        return np.zeros(len(query_xy), dtype=np.int32)
    centroids = {}
    for tid in team_ids:
        pts = player_xy[player_team_id == tid]
        if len(pts) == 0:
            continue
        centroids[int(tid)] = pts.mean(axis=0)
    if not centroids:
        return np.zeros(len(query_xy), dtype=np.int32)
    out = []
    for xy in query_xy:
        best_tid = min(
            centroids.keys(),
            key=lambda tid: float(np.linalg.norm(xy - centroids[tid])),
        )
        out.append(best_tid)
    return np.asarray(out, dtype=np.int32)


class TeamClassifier:
    """SigLIP features → UMAP(3) → KMeans(2), matching Roboflow sports."""

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        import umap
        from sklearn.cluster import KMeans

        self.device = device
        self.batch_size = batch_size
        self._torch = torch
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.features_model.eval()
        # Image-only processor — avoids SentencePiece/tokenizer dependency.
        self.processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3, random_state=0)
        self.cluster_model = KMeans(n_clusters=2, n_init=20, random_state=0)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def extract_features(self, crops: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        if not crops:
            return np.zeros((0, 768), dtype=np.float32)
        pil_crops = [
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops
        ]
        data = []
        with self._torch.no_grad():
            for batch in create_batches(pil_crops, self.batch_size):
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.features_model(**inputs)
                embeddings = (
                    self._torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                )
                data.append(embeddings)
        return np.concatenate(data, axis=0)

    def fit(self, crops: list[np.ndarray]) -> None:
        if len(crops) < 4:
            raise ValueError("need at least 4 crops to fit two-team classifier")
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        self._fitted = True

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.array([], dtype=np.int32)
        if not self._fitted:
            raise RuntimeError("TeamClassifier.fit() must be called first")
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections).astype(np.int32)


def _is_dark_crop(frame_bgr: np.ndarray, bbox: Sequence[float]) -> bool:
    """Referee / dark GK — must not train or vote in the two-team cluster."""
    from football_analytics.analytics.kit_descriptor import (
        is_dark_kit_fractions,
        kit_feature_from_frame,
    )

    feat, _ = kit_feature_from_frame(frame_bgr, bbox)
    if feat is None:
        return False
    return bool(is_dark_kit_fractions(feat))


def fit_team_classifier_from_video(
    video_path: str,
    tracks,
    *,
    stride: int = 30,
    max_crops: int = 400,
    device: str = "cuda",
    batch_size: int = 32,
    min_box_area: float = 400.0,
    exclude_dark: bool = True,
) -> TeamClassifier:
    """Collect person crops evenly across the clip and fit SigLIP team classifier.

    Unlike a first-N scan (which biases to one half of the pitch), we gather
    candidates on ``stride`` then subsample uniformly up to ``max_crops``.
    Dark kits are dropped from the fit set (sports-main trains on players only).
    """
    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
    by_frame = {int(fid): g for fid, g in person.groupby("frame_id")}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    candidates: list[np.ndarray] = []
    frame_id = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % stride != 0:
            continue
        g = by_frame.get(frame_id)
        if g is None:
            continue
        for row in g.itertuples(index=False):
            area = (float(row.bbox_x2) - float(row.bbox_x1)) * (
                float(row.bbox_y2) - float(row.bbox_y1)
            )
            if area < min_box_area:
                continue
            bbox = (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2)
            if exclude_dark and _is_dark_crop(frame, bbox):
                continue
            crop = crop_person_bgr(frame, bbox)
            if crop is None:
                continue
            candidates.append(crop)
    cap.release()
    if len(candidates) < 8:
        raise RuntimeError(f"too few crops for SigLIP team fit: {len(candidates)}")
    if len(candidates) > max_crops:
        idx = np.linspace(0, len(candidates) - 1, max_crops, dtype=int)
        crops = [candidates[int(i)] for i in idx]
    else:
        crops = candidates
    clf = TeamClassifier(device=device, batch_size=batch_size)
    clf.fit(crops)
    return clf


def assign_teams_with_classifier(
    video_path: str,
    tracks,
    classifier: TeamClassifier,
    *,
    predict_stride: int = 5,
    min_votes: int = 2,
    exclude_dark: bool = True,
) -> tuple[dict[int, tuple[int, float]], set[int]]:
    """Return (track→(team,conf), dark_track_ids).

    Dark crops never vote; tracks that are mostly dark become referee/unknown.
    """
    person = tracks[tracks["object_type"].eq("person")] if "object_type" in tracks.columns else tracks
    by_frame = {int(fid): g for fid, g in person.groupby("frame_id")}
    votes: dict[int, list[int]] = defaultdict(list)
    dark_hits: dict[int, int] = defaultdict(int)
    seen: dict[int, int] = defaultdict(int)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_id = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % predict_stride != 0:
            continue
        g = by_frame.get(frame_id)
        if g is None or g.empty:
            continue
        crop_list: list[np.ndarray] = []
        tid_list: list[int] = []
        for row in g.itertuples(index=False):
            tid = int(row.track_id)
            bbox = (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2)
            seen[tid] += 1
            if exclude_dark and _is_dark_crop(frame, bbox):
                dark_hits[tid] += 1
                continue
            crop = crop_person_bgr(frame, bbox)
            if crop is None:
                continue
            crop_list.append(crop)
            tid_list.append(tid)
        if not crop_list:
            continue
        preds = classifier.predict(crop_list)
        for tid, pred in zip(tid_list, preds.tolist()):
            votes[int(tid)].append(int(pred))
    cap.release()

    dark_tracks: set[int] = set()
    for tid, n in seen.items():
        if n > 0 and dark_hits[tid] / float(n) >= 0.55:
            dark_tracks.add(int(tid))

    out: dict[int, tuple[int, float]] = {}
    for tid, v in votes.items():
        if tid in dark_tracks:
            continue
        if len(v) < min_votes:
            continue
        counts = Counter(v)
        team, n = counts.most_common(1)[0]
        conf = float(n) / float(len(v))
        out[int(tid)] = (int(team), conf)
    return out, dark_tracks
