"""Resolution-independent streaming renderer for canonical analytics artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


TEAM_COLORS = {
    "team_0": (255, 90, 40),
    "team_1": (40, 90, 255),
    "unknown": (170, 170, 170),
}


def _rows_by_frame(frame: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if frame.empty or "frame_id" not in frame:
        return rows
    for item in frame.to_dict("records"):
        rows[int(item["frame_id"])].append(item)
    return rows


def _nullable_text(value: Any, fallback: str = "?") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    return str(value)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class _LabelStabilizer:
    """EMA-smooth label anchors so overlay text does not jitter with bbox noise."""

    def __init__(self, *, alpha: float = 0.28, hold_frames: int = 6) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.hold_frames = max(0, int(hold_frames))
        self._pos: dict[int, tuple[float, float]] = {}
        self._box: dict[int, tuple[float, float, float, float]] = {}
        self._text: dict[int, str] = {}
        self._color: dict[int, tuple[int, int, int]] = {}
        self._last_frame: dict[int, int] = {}

    def observe(
        self,
        display_id: int,
        *,
        frame_id: int,
        box: tuple[int, int, int, int],
        text: str,
        color: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[int, int], str, tuple[int, int, int]]:
        x1, y1, x2, y2 = (float(v) for v in box)
        cx = 0.5 * (x1 + x2)
        top = y1
        if display_id in self._pos:
            pcx, ptop = self._pos[display_id]
            a = self.alpha
            cx = a * cx + (1.0 - a) * pcx
            top = a * top + (1.0 - a) * ptop
            px1, py1, px2, py2 = self._box[display_id]
            x1 = a * x1 + (1.0 - a) * px1
            y1 = a * y1 + (1.0 - a) * py1
            x2 = a * x2 + (1.0 - a) * px2
            y2 = a * y2 + (1.0 - a) * py2
            # Keep text identity sticky within hold window (avoid P#/A/B flicker).
            if text != self._text.get(display_id) and (
                frame_id - self._last_frame.get(display_id, frame_id) <= 2
            ):
                # Prefer previous short label if only speed digit changed; for ID keep new.
                prev = self._text.get(display_id, text)
                if prev.split(" ", 1)[0] == text.split(" ", 1)[0]:
                    text = prev
        self._pos[display_id] = (cx, top)
        self._box[display_id] = (x1, y1, x2, y2)
        self._text[display_id] = text
        self._color[display_id] = color
        self._last_frame[display_id] = frame_id
        label_xy = (int(round(cx)), int(round(top)))
        smooth_box = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )
        return smooth_box, label_xy, text, color

    def ghosts(self, frame_id: int, active: set[int]) -> list[tuple[int, tuple[int, int], str, tuple[int, int, int]]]:
        """Briefly keep labels for tracks that dropped for a few frames."""
        out: list[tuple[int, tuple[int, int], str, tuple[int, int, int]]] = []
        for display_id, last in list(self._last_frame.items()):
            if display_id in active:
                continue
            age = frame_id - last
            if age <= 0 or age > self.hold_frames:
                continue
            cx, top = self._pos[display_id]
            out.append(
                (
                    display_id,
                    (int(round(cx)), int(round(top))),
                    self._text[display_id],
                    self._color[display_id],
                )
            )
        return out


def _nms_person_tracks(
    tracks: list[dict[str, Any]],
    display_ids: dict[int, int],
    *,
    preferred_displays: set[int] | None = None,
    iou_thresh: float = 0.55,
) -> list[dict[str, Any]]:
    """Keep one box per overlapping cluster; prefer continuity then stable id."""
    preferred = preferred_displays or set()
    scored = []
    for track in tracks:
        tid = int(track["track_id"])
        display = display_ids.get(tid, 10_000 + tid)
        box = (
            int(track["bbox_x1"]),
            int(track["bbox_y1"]),
            int(track["bbox_x2"]),
            int(track["bbox_y2"]),
        )
        scored.append((0 if display in preferred else 1, display, tid, box, track))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    kept: list[dict[str, Any]] = []
    kept_boxes: list[tuple[int, int, int, int]] = []
    kept_display: list[int] = []
    for _pref, display, _tid, box, track in scored:
        suppress = False
        for kb, kd in zip(kept_boxes, kept_display):
            iou = _iou(box, kb)
            if kd == display and iou >= 0.20:
                suppress = True
                break
            if kd != display and iou >= max(iou_thresh, 0.55):
                suppress = True
                break
        if suppress:
            continue
        kept.append(track)
        kept_boxes.append(box)
        kept_display.append(display)
    return kept


def _draw_label(
    image: np.ndarray,
    *,
    text: str,
    anchor_xy: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    cx, top = anchor_xy
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Center the label above the player; clamp inside frame.
    tx = int(np.clip(cx - tw // 2, 2, max(2, image.shape[1] - tw - 2)))
    ty = int(np.clip(top - 8, th + 4, image.shape[0] - 4))
    cv2.rectangle(
        image,
        (tx - 3, ty - th - 5),
        (tx + tw + 3, ty + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_analytics_video(
    input_video: Path,
    output_video: Path,
    tracks: pd.DataFrame,
    identities: pd.DataFrame,
    ball_state: pd.DataFrame,
    possession: pd.DataFrame,
    player_metrics: pd.DataFrame,
    camera_motion: pd.DataFrame,
    calibration: pd.DataFrame,
    layers: Iterable[str],
    display_id_by_track: dict[int, int] | None = None,
    team_by_track: dict[int, str] | None = None,
    frame_display_ids: dict[tuple[int, int], int] | None = None,
) -> dict[str, Any]:
    """Render frame-by-frame; video frames are never accumulated in memory."""
    layer_set = set(layers)
    display_ids = display_id_by_track or {}
    locked_teams = team_by_track or {}
    frame_ids = frame_display_ids or {}
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output writer: {output_video}")

    track_rows = _rows_by_frame(tracks)
    identity_rows = _rows_by_frame(identities)
    ball_rows = _rows_by_frame(ball_state)
    possession_rows = _rows_by_frame(possession)
    metric_rows = _rows_by_frame(player_metrics)
    motion_rows = _rows_by_frame(camera_motion)
    calibration_rows = _rows_by_frame(calibration)
    identity_lookup = {
        (int(row["frame_id"]), int(row["track_id"])): row
        for frame_rows in identity_rows.values()
        for row in frame_rows
        if row.get("track_id") is not None
    }

    stabilizer = _LabelStabilizer(alpha=0.22, hold_frames=8)
    preferred_displays: set[int] = set()

    frame_id = 0
    written = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        font_scale = max(0.45, min(width, height) / 1300.0)
        thickness = max(1, int(round(min(width, height) / 700.0)))
        raw_tracks = [
            t
            for t in track_rows.get(frame_id, [])
            if str(t.get("object_type") or "person") in {"person", "player"}
        ]
        # Prefer overlay slot ids for NMS continuity when present.
        nms_display = {
            int(t["track_id"]): int(
                frame_ids.get(
                    (frame_id, int(t["track_id"])),
                    display_ids.get(int(t["track_id"]), 10_000 + int(t["track_id"])),
                )
            )
            for t in raw_tracks
        }
        person_tracks = _nms_person_tracks(
            raw_tracks, nms_display, preferred_displays=preferred_displays
        )
        active_displays: set[int] = set()
        for track in person_tracks:
            tid = int(track["track_id"])
            display = frame_ids.get((frame_id, tid), display_ids.get(tid))
            if display is None:
                continue
            display = int(display)
            ident = identity_lookup.get((frame_id, tid), {})
            team = locked_teams.get(tid) or _nullable_text(ident.get("team_id"), "unknown")
            color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
            raw_box = (
                int(track["bbox_x1"]),
                int(track["bbox_y1"]),
                int(track["bbox_x2"]),
                int(track["bbox_y2"]),
            )
            labels: list[str] = []
            if "track_id" in layer_set:
                labels.append(f"P{display}")
            if "team" in layer_set and team not in {"unknown", "?"}:
                labels.append("A" if team.endswith("0") else "B" if team.endswith("1") else team)
            text = " ".join(labels) if labels else f"P{display}"
            smooth_box, anchor, text, color = stabilizer.observe(
                display,
                frame_id=frame_id,
                box=raw_box,
                text=text,
                color=color,
            )
            active_displays.add(display)
            x1, y1, x2, y2 = smooth_box
            foot = (int((x1 + x2) / 2), y2)
            axes = (max(8, int((x2 - x1) * 0.55)), max(4, int((x2 - x1) * 0.18)))
            cv2.ellipse(image, foot, axes, 0, 200, 340, color, thickness + 1)
            _draw_label(
                image,
                text=text,
                anchor_xy=anchor,
                color=color,
                font_scale=font_scale,
                thickness=thickness,
            )

        for _did, anchor, text, color in stabilizer.ghosts(frame_id, active_displays):
            _draw_label(
                image,
                text=text,
                anchor_xy=anchor,
                color=color,
                font_scale=font_scale,
                thickness=thickness,
            )
        preferred_displays = active_displays

        if "ball" in layer_set:
            for ball in ball_rows.get(frame_id, []):
                x, y = ball.get("ball_x_pixel"), ball.get("ball_y_pixel")
                if x is None or y is None or pd.isna(x) or pd.isna(y):
                    continue
                p = (int(x), int(y))
                radius = max(6, int(min(width, height) * 0.008))
                points = np.array(
                    [[p[0], p[1]], [p[0] - radius, p[1] - 2 * radius], [p[0] + radius, p[1] - 2 * radius]]
                )
                cv2.drawContours(image, [points], 0, (0, 255, 255), -1)

        panel_h = max(52, int(height * 0.12))
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        lines: list[str] = []
        if "possession" in layer_set and possession_rows.get(frame_id):
            row = possession_rows[frame_id][0]
            lines.append(
                f"Possession: {_nullable_text(row.get('possession_state'), 'unknown')} "
                f"({float(row.get('confidence') or 0):.2f})"
            )
        if "camera_motion" in layer_set and motion_rows.get(frame_id):
            row = motion_rows[frame_id][0]
            lines.append(
                f"Camera dx={float(row.get('dx_pixel') or 0):.1f} "
                f"dy={float(row.get('dy_pixel') or 0):.1f} valid={bool(row.get('valid'))}"
            )
        if "calibration" in layer_set:
            row = calibration_rows.get(frame_id, [{}])[0]
            lines.append(
                "Calibration: valid"
                if row.get("valid")
                else f"Calibration: unavailable ({_nullable_text(row.get('invalid_reason'))})"
            )
        if "warnings" in layer_set and not calibration_rows.get(frame_id, [{}])[0].get("valid"):
            lines.append("Spatial speed/distance/team shape withheld")
        for index, line in enumerate(lines[:3]):
            cv2.putText(
                image,
                line,
                (int(width * 0.015), int((index + 1) * panel_h / 3.6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        writer.write(image)
        written += 1
        frame_id += 1

    writer.release()
    capture.release()
    if written == 0:
        raise RuntimeError("Analytics renderer wrote zero frames")
    return {"frames": written, "fps": fps, "width": width, "height": height}



def render_tactical_preview(
    output_video: Path,
    input_fps: float,
    frame_count: int,
    game_state: pd.DataFrame,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    display_id_by_track: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Render a tactical pitch; invalid frames clearly state no calibration."""
    width, height = 840, 544
    display_ids = display_id_by_track or {}
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), input_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create tactical preview: {output_video}")
    grouped = _rows_by_frame(game_state)
    valid_frames = 0
    stabilizer = _LabelStabilizer(alpha=0.35, hold_frames=4)
    for frame_id in range(frame_count):
        image = np.full((height, width, 3), (35, 110, 35), dtype=np.uint8)
        margin = 32
        cv2.rectangle(image, (margin, margin), (width - margin, height - margin), (255, 255, 255), 2)
        cv2.line(image, (width // 2, margin), (width // 2, height - margin), (255, 255, 255), 2)
        cv2.circle(image, (width // 2, height // 2), 52, (255, 255, 255), 2)
        rows = [row for row in grouped.get(frame_id, []) if row.get("valid")]
        active: set[int] = set()
        if rows:
            valid_frames += 1
            for row in rows:
                x = margin + int(float(row["x_field"]) / pitch_length_m * (width - 2 * margin))
                y = margin + int(float(row["y_field"]) / pitch_width_m * (height - 2 * margin))
                color = TEAM_COLORS.get(_nullable_text(row.get("team_id"), "unknown"), (180, 180, 180))
                tid = row.get("track_id")
                if tid is None or (isinstance(tid, float) and np.isnan(tid)):
                    cv2.circle(image, (x, y), 7, color, -1)
                    continue
                display = display_ids.get(int(tid), int(tid))
                active.add(int(display))
                smooth_box, anchor, text, color = stabilizer.observe(
                    int(display),
                    frame_id=frame_id,
                    box=(x - 6, y - 6, x + 6, y + 6),
                    text=f"P{display}",
                    color=color,
                )
                cx = (smooth_box[0] + smooth_box[2]) // 2
                cy = (smooth_box[1] + smooth_box[3]) // 2
                cv2.circle(image, (cx, cy), 7, color, -1)
                _draw_label(
                    image,
                    text=text,
                    anchor_xy=(anchor[0], anchor[1] - 2),
                    color=(255, 255, 255),
                    font_scale=0.35,
                    thickness=1,
                )
        else:
            cv2.putText(
                image,
                "NO VALID CALIBRATION - POSITIONS WITHHELD",
                (90, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(image)
    writer.release()
    return {"frames": frame_count, "valid_frames": valid_frames}
