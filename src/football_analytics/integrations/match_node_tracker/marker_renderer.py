"""Renderer markers adapted from Match-node-tracker custom_markers / draw_speed.

Upstream source:
  - third_party/authorized/match-node-tracker/custom_markers.py
  - third_party/authorized/match-node-tracker/speed_tracker.py (draw_speed only)
Upstream commit: 2777aa3f1e9cc563eba07a675cebdf4bfd9306bf
Changes vs upstream:
  - pure overlay; never mutates tracks / detections / possession state
  - optional feature flag
  - speed labels only if caller supplies speeds (no upstream speed formula)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

UPSTREAM_COMMIT = "2777aa3f1e9cc563eba07a675cebdf4bfd9306bf"


@dataclass
class MatchNodeMarkerConfig:
    enabled: bool = False
    draw_ball: bool = True
    draw_referee: bool = True
    draw_player_triangle: bool = True
    draw_possession_bar: bool = True
    draw_speed_label: bool = False
    scale: int = 2


class MatchNodeMarkerRenderer:
    """Draw-only markers. Tracking/analytics inputs are read-only."""

    def __init__(self, config: MatchNodeMarkerConfig | None = None) -> None:
        self.config = config or MatchNodeMarkerConfig()

    def draw(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        labels: list[str],
        team_ids: list[int] | None = None,
        team_colors: list[tuple[int, int, int]] | None = None,
        speeds: list[float | None] | None = None,
        possession_pct: list[float] | None = None,
    ) -> np.ndarray:
        if not self.config.enabled:
            return img
        out = img
        team_ids = team_ids or [-1] * len(boxes)
        team_colors = team_colors or [(220, 80, 60), (60, 80, 220)]
        scale = self.config.scale
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            lbl = (labels[i] if i < len(labels) else "").lower()
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
            if ("ball" in lbl or "football" in lbl) and self.config.draw_ball:
                cv2.fillPoly(
                    out,
                    [np.array([[xc, yc - 10], [xc - 14, yc - 30], [xc + 14, yc - 30]])],
                    (0, 255, 255),
                )
                cv2.circle(out, (xc, yc), 8, (0, 255, 255), 2, cv2.LINE_AA)
            elif "referee" in lbl and self.config.draw_referee:
                ax = min(max(14, (x2 - x1) // 2 + 10), 45)
                ay = min(max(7, (x2 - x1) // 6 + 3), 20)
                cv2.ellipse(out, (xc, y2), (ax + 5, ay + 3), 0, -45, 235, (0, 140, 140), 7, cv2.LINE_AA)
                cv2.ellipse(out, (xc, y2), (ax, ay), 0, -45, 235, (0, 255, 255), 2, cv2.LINE_AA)
            elif self.config.draw_player_triangle:
                tid = team_ids[i] if i < len(team_ids) else -1
                if tid >= 0:
                    color = team_colors[tid % len(team_colors)]
                    pts = np.array(
                        [
                            [xc, y1 - 5 * scale],
                            [xc - 10 * scale, y1 - 20 * scale],
                            [xc + 10 * scale, y1 - 20 * scale],
                        ]
                    )
                    cv2.fillPoly(out, [pts], color)
            if (
                self.config.draw_speed_label
                and speeds is not None
                and i < len(speeds)
                and speeds[i] is not None
                and speeds[i] >= 1.0
            ):
                txt = f"{speeds[i]:.1f} km/h"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                bw, bh = tw + 12, th + 8
                bx, by = (x1 + x2) // 2 - bw // 2, y2 + 10
                cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
                cv2.putText(
                    out,
                    txt,
                    (bx + 6, by + bh - 5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        if self.config.draw_possession_bar and possession_pct and len(possession_pct) >= 2:
            h, w = out.shape[:2]
            bw, bh, by, sx = int(w * 0.4), 20, h - 50, int(w * 0.3)
            w0 = int(bw * possession_pct[0] / 100)
            cv2.rectangle(out, (sx, by), (sx + w0, by + bh), team_colors[0], -1)
            cv2.rectangle(out, (sx + w0, by), (sx + bw, by + bh), team_colors[1], -1)
            cv2.putText(
                out,
                f"{int(possession_pct[0])}%",
                (sx - 50, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                team_colors[0],
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                f"{int(possession_pct[1])}%",
                (sx + bw + 10, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                team_colors[1],
                2,
                cv2.LINE_AA,
            )
        return out
