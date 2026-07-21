"""Field-occupancy heatmaps from calibrated positions only.

Cells accumulate dwell time (not raw sample counts) so variable frame rates do
not distort the map. Samples are used only when calibration is valid, the
position is inside the modelled pitch bounds, and the frame is not a replay;
excluded samples contribute nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class HeatmapConfig:
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    bins_x: int = 21
    bins_y: int = 14
    #: Time gaps longer than this break dwell accumulation (discontinuity).
    max_gap_ms: float = 1500.0

    def __post_init__(self) -> None:
        if self.pitch_length_m <= 0 or self.pitch_width_m <= 0:
            raise ValueError("pitch dimensions must be positive")
        if self.bins_x < 1 or self.bins_y < 1:
            raise ValueError("bins must be >= 1")


@dataclass(frozen=True)
class HeatmapSample:
    timestamp_ms: float
    x_field: float | None
    y_field: float | None
    calibration_valid: bool = True
    replay: bool = False


@dataclass(frozen=True)
class Heatmap:
    """Row-major grid of dwell seconds: ``grid[iy][ix]``."""

    grid: tuple[tuple[float, ...], ...]
    total_dwell_s: float
    used_samples: int
    excluded_samples: int

    @property
    def normalized(self) -> tuple[tuple[float, ...], ...]:
        if self.total_dwell_s <= 0.0:
            return self.grid
        return tuple(
            tuple(cell / self.total_dwell_s for cell in row) for row in self.grid
        )


def _usable(sample: HeatmapSample, cfg: HeatmapConfig) -> bool:
    return (
        not sample.replay
        and sample.calibration_valid
        and sample.x_field is not None
        and sample.y_field is not None
        and math.isfinite(float(sample.x_field))
        and math.isfinite(float(sample.y_field))
        and 0.0 <= float(sample.x_field) <= cfg.pitch_length_m
        and 0.0 <= float(sample.y_field) <= cfg.pitch_width_m
    )


def _cell(sample: HeatmapSample, cfg: HeatmapConfig) -> tuple[int, int]:
    ix = min(int(float(sample.x_field) / cfg.pitch_length_m * cfg.bins_x), cfg.bins_x - 1)
    iy = min(int(float(sample.y_field) / cfg.pitch_width_m * cfg.bins_y), cfg.bins_y - 1)
    return ix, iy


def compute_heatmap(
    samples: Iterable[HeatmapSample],
    config: HeatmapConfig | None = None,
) -> Heatmap:
    """Accumulate dwell time per cell for one track's samples."""
    cfg = config or HeatmapConfig()
    ordered: Sequence[HeatmapSample] = sorted(samples, key=lambda s: s.timestamp_ms)
    grid = [[0.0] * cfg.bins_x for _ in range(cfg.bins_y)]
    total = 0.0
    used = 0
    excluded = 0

    previous: HeatmapSample | None = None
    for sample in ordered:
        if not _usable(sample, cfg):
            excluded += 1
            previous = None
            continue
        used += 1
        if previous is not None:
            gap_ms = sample.timestamp_ms - previous.timestamp_ms
            if 0.0 < gap_ms <= cfg.max_gap_ms:
                # Split the elapsed time between the two cells.
                dwell_s = gap_ms / 1000.0
                for anchor, share in ((previous, 0.5), (sample, 0.5)):
                    ix, iy = _cell(anchor, cfg)
                    grid[iy][ix] += dwell_s * share
                total += dwell_s
        previous = sample

    return Heatmap(
        grid=tuple(tuple(row) for row in grid),
        total_dwell_s=total,
        used_samples=used,
        excluded_samples=excluded,
    )
