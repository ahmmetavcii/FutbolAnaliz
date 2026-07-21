"""Football-ball recovery: ROI / Kalman / short-gap fill on top of YOLO detections.

Does not invent long-horizon ball positions. Secondary ROI search can use the
same Ultralytics weights at higher imgsz on a crop around the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BallRecoveryConfig:
    ball_class_id: int = 32
    roi_conf: float = 0.12
    roi_imgsz: int = 320
    roi_half_size: int = 160
    max_interp_frames: int = 5
    max_interp_seconds: float = 0.25
    max_pixel_speed: float = 2500.0
    kalman_process_noise: float = 80.0
    kalman_measurement_noise: float = 8.0
    enable_roi_search: bool = True
    enable_optical_flow: bool = True
    model_path: str | None = None
    device: str | int = 0


class _Kalman2D:
    def __init__(self, q: float, r: float) -> None:
        self.x = np.zeros(4, dtype=np.float64)  # x,y,vx,vy
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.q = q
        self.r = r
        self.initialized = False

    def predict(self, dt: float) -> tuple[float, float]:
        if not self.initialized:
            return float("nan"), float("nan")
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        Q = np.eye(4, dtype=np.float64) * self.q * max(dt, 1e-3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0]), float(self.x[1])

    def update(self, zx: float, zy: float) -> None:
        if not self.initialized:
            self.x[:] = [zx, zy, 0.0, 0.0]
            self.initialized = True
            return
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        z = np.array([zx, zy], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * self.r
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


def _raw_detection_mask(ball_state: pd.DataFrame) -> pd.Series:
    if "visibility_state" not in ball_state.columns:
        return pd.Series(False, index=ball_state.index)
    return ball_state["visibility_state"].astype(str).eq("detected")


def enhance_ball_state(
    ball_state: pd.DataFrame,
    *,
    video_path: Path | None = None,
    config: BallRecoveryConfig | None = None,
    model: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Improve coverage with Kalman prediction + optional ROI YOLO + short interp."""
    cfg = config or BallRecoveryConfig()
    if ball_state is None or ball_state.empty:
        return ball_state, _empty_coverage()

    frame = ball_state.sort_values("frame_id").reset_index(drop=True).copy()
    raw_mask = _raw_detection_mask(frame)
    raw_coverage = float(raw_mask.mean()) if len(frame) else 0.0

    kf = _Kalman2D(cfg.kalman_process_noise, cfg.kalman_measurement_noise)
    prev_gray = None
    prev_pt = None
    false_positive_candidates = 0
    roi_hits = 0
    flow_hits = 0
    cap = None
    if cfg.enable_roi_search and video_path is not None and Path(video_path).is_file():
        cap = cv2.VideoCapture(str(video_path))

    yolo = model
    if yolo is None and cfg.enable_roi_search and cfg.model_path:
        try:
            from ultralytics import YOLO

            yolo = YOLO(str(cfg.model_path))
        except Exception:  # noqa: BLE001
            yolo = None

    xs = frame["ball_x_pixel"].to_numpy(dtype=float)
    ys = frame["ball_y_pixel"].to_numpy(dtype=float)
    states = frame["visibility_state"].astype(str).tolist()
    confs = (
        frame["detection_confidence"].fillna(0.0).to_numpy(dtype=float)
        if "detection_confidence" in frame.columns
        else np.zeros(len(frame))
    )
    timestamps = frame["timestamp_ms"].to_numpy(dtype=float)
    sources = ["raw"] * len(frame)

    last_raw_index = -10_000
    for i in range(len(frame)):
        dt = 0.04 if i == 0 else max(1e-3, (timestamps[i] - timestamps[i - 1]) / 1000.0)
        pred_x, pred_y = kf.predict(dt)
        detected = states[i] == "detected" and np.isfinite(xs[i]) and np.isfinite(ys[i])
        already_tracked = states[i] in {
            "predicted",
            "occluded_short",
            "airborne",
        } and np.isfinite(xs[i]) and np.isfinite(ys[i])

        if detected or already_tracked:
            if detected and kf.initialized and np.isfinite(pred_x):
                speed = float(np.hypot(xs[i] - pred_x, ys[i] - pred_y) / dt)
                if speed > cfg.max_pixel_speed * 1.5:
                    false_positive_candidates += 1
                    detected = False
                    already_tracked = False
                    states[i] = "unknown"
                    xs[i] = np.nan
                    ys[i] = np.nan
            if detected or already_tracked:
                kf.update(float(xs[i]), float(ys[i]))
                prev_pt = (float(xs[i]), float(ys[i]))
                sources[i] = "raw_detection" if detected else f"ball_state_{states[i]}"
                if detected or states[i] in {"detected", "airborne"}:
                    last_raw_index = i
                elif already_tracked and (i - last_raw_index) <= cfg.max_interp_frames:
                    pass
                else:
                    # Preserve existing short prediction from ball_state
                    last_raw_index = max(last_raw_index, i - cfg.max_interp_frames)
                if cap is not None and detected:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame.loc[i, "frame_id"]))
                    ok, bgr = cap.read()
                    if ok:
                        prev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                continue

        frames_since_raw = i - last_raw_index
        allow_short_fill = 0 < frames_since_raw <= cfg.max_interp_frames

        # Missing detection: try optical flow then ROI YOLO (short window only)
        recovered = False
        if (
            allow_short_fill
            and cfg.enable_optical_flow
            and prev_gray is not None
            and prev_pt is not None
            and cap is not None
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame.loc[i, "frame_id"]))
            ok, bgr = cap.read()
            if ok:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pts = np.array([[prev_pt]], dtype=np.float32)
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                if st is not None and int(st[0, 0]) == 1:
                    fx, fy = float(nxt[0, 0, 0]), float(nxt[0, 0, 1])
                    if (not np.isfinite(pred_x)) or float(
                        np.hypot(fx - pred_x, fy - pred_y)
                    ) < cfg.roi_half_size:
                        xs[i], ys[i] = fx, fy
                        states[i] = "predicted"
                        confs[i] = max(float(confs[i]), 0.20)
                        sources[i] = "optical_flow"
                        kf.update(fx, fy)
                        prev_pt = (fx, fy)
                        prev_gray = gray
                        recovered = True
                        flow_hits += 1

        if (
            not recovered
            and allow_short_fill
            and yolo is not None
            and cap is not None
            and np.isfinite(pred_x)
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame.loc[i, "frame_id"]))
            ok, bgr = cap.read()
            if ok:
                h, w = bgr.shape[:2]
                x0 = int(max(0, pred_x - cfg.roi_half_size))
                y0 = int(max(0, pred_y - cfg.roi_half_size))
                x1 = int(min(w, pred_x + cfg.roi_half_size))
                y1 = int(min(h, pred_y + cfg.roi_half_size))
                crop = bgr[y0:y1, x0:x1]
                if crop.size > 0:
                    try:
                        results = yolo.predict(
                            source=crop,
                            conf=cfg.roi_conf,
                            imgsz=cfg.roi_imgsz,
                            classes=[cfg.ball_class_id],
                            verbose=False,
                        )
                        best = None
                        for r in results:
                            if r.boxes is None:
                                continue
                            for box in r.boxes:
                                conf = float(box.conf[0])
                                xyxy = box.xyxy[0].tolist()
                                cx = x0 + (xyxy[0] + xyxy[2]) / 2.0
                                cy = y0 + (xyxy[1] + xyxy[3]) / 2.0
                                if best is None or conf > best[0]:
                                    best = (conf, cx, cy)
                        if best is not None:
                            xs[i], ys[i] = best[1], best[2]
                            states[i] = "detected"
                            confs[i] = best[0]
                            sources[i] = "roi_yolo"
                            kf.update(best[1], best[2])
                            prev_pt = (best[1], best[2])
                            prev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                            last_raw_index = i
                            recovered = True
                            roi_hits += 1
                    except Exception:  # noqa: BLE001
                        pass

        if not recovered and allow_short_fill and kf.initialized and np.isfinite(pred_x):
            xs[i], ys[i] = pred_x, pred_y
            states[i] = "predicted"
            confs[i] = min(0.25, max(float(confs[i]), 0.15))
            sources[i] = "kalman_predict"
            prev_pt = (pred_x, pred_y)
        elif not recovered:
            # Keep existing coords only if still finite from input; else unknown
            if not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
                xs[i] = np.nan
                ys[i] = np.nan
                states[i] = "unknown"
                sources[i] = "missing"
                confs[i] = 0.0
            if frames_since_raw > cfg.max_interp_frames:
                kf.initialized = False
                prev_pt = None
                prev_gray = None


    # Enforce max interpolation run length
    run = 0
    for i in range(len(frame)):
        is_interp = sources[i] in {"kalman_predict", "optical_flow"} and states[i] != "detected"
        if is_interp:
            run += 1
            if run > cfg.max_interp_frames:
                xs[i] = np.nan
                ys[i] = np.nan
                states[i] = "unknown"
                sources[i] = "interp_capped"
                confs[i] = 0.0
                run = 0
                kf.initialized = False
        else:
            run = 0 if states[i] == "unknown" else 0 if sources[i] == "interp_capped" else 0
            if states[i] == "detected" or sources[i] == "roi_yolo":
                run = 0

    # Time-based longest missing
    missing = []
    start = None
    for i in range(len(frame)):
        visible = states[i] in {"detected", "predicted", "occluded_short", "airborne"} and np.isfinite(
            xs[i]
        )
        if not visible:
            if start is None:
                start = timestamps[i]
        else:
            if start is not None:
                missing.append(timestamps[i] - start)
                start = None
    if start is not None:
        missing.append(timestamps[-1] - start)

    frame["ball_x_pixel"] = xs
    frame["ball_y_pixel"] = ys
    frame["visibility_state"] = states
    if "detection_confidence" in frame.columns:
        frame["detection_confidence"] = confs
    frame["ball_source"] = sources
    # Clear field coords when pixel unknown (avoid stale)
    if "ball_x_field" in frame.columns:
        unknown = ~np.isfinite(xs)
        frame.loc[unknown, "ball_x_field"] = np.nan
        frame.loc[unknown, "ball_y_field"] = np.nan
        frame.loc[unknown, "valid"] = False

    tracked = frame["visibility_state"].isin(["detected", "predicted", "occluded_short", "airborne"]) & np.isfinite(
        frame["ball_x_pixel"]
    )
    interp = frame["ball_source"].isin(["kalman_predict", "optical_flow"]) & tracked
    coverage = {
        "raw_detection_coverage": round(raw_coverage, 4),
        "tracked_coverage": round(float(tracked.mean()), 4) if len(frame) else 0.0,
        "interpolated_coverage": round(float(interp.mean()), 4) if len(frame) else 0.0,
        "longest_missing_interval_ms": float(max(missing)) if missing else 0.0,
        "false_positive_candidates": int(false_positive_candidates),
        "roi_hits": int(roi_hits),
        "optical_flow_hits": int(flow_hits),
        "frames": int(len(frame)),
    }
    if cap is not None:
        cap.release()
    return frame, coverage


def _empty_coverage() -> dict[str, Any]:
    return {
        "raw_detection_coverage": 0.0,
        "tracked_coverage": 0.0,
        "interpolated_coverage": 0.0,
        "longest_missing_interval_ms": 0.0,
        "false_positive_candidates": 0,
        "roi_hits": 0,
        "optical_flow_hits": 0,
        "frames": 0,
    }


def write_ball_detector_adapter_spec(path: Path) -> Path:
    """Document how to plug a football-specific ball detector / fine-tune."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Football ball detector adapter

Production primary path: football-specific YOLO (`football_ball` config /
`yolo-sn-ball-opt.pt`). COCO `sports ball` (class 32) is fallback only.

To fine-tune or swap another football-specific detector:

1. Train YOLO on football-ball crops (small object, motion blur, occlusion).
2. Point `ball_trajectory.ball_model_path` at the new weights.
3. Keep `ball_class_ids: [0]` if the fine-tuned model is single-class.
4. ROI recovery (`enhance_ball_state`) reuses the same weights at higher
   effective resolution around the Kalman prediction.

Do not invent detections outside ROI + physical gates.
""",
        encoding="utf-8",
    )
    return path
