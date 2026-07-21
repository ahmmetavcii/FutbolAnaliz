"""Panel Turkish label and tab structure tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_analytics.panel.opta_labels import (  # noqa: E402
    PLAYER_COLUMN_TR,
    TAB_NAMES,
    format_value,
    quality_label,
    rename_frame,
)

PANEL_PATH = PROJECT_ROOT / "apps" / "full_match_panel.py"
PANEL_SOURCE = PANEL_PATH.read_text(encoding="utf-8")


def test_panel_turkish_tabs_present():
    for name in TAB_NAMES:
        assert name in (
            "Genel Özet",
            "Oyuncular",
            "Takımlar",
            "Paslar",
            "Mücadeleler",
            "Dribbling",
            "Savunma Aksiyonları",
            "Fiziksel Veriler",
            "Isı Haritaları",
            "Maç Olayları",
            "Veri Kalitesi",
        )
    assert "st.tabs(TAB_NAMES)" in PANEL_SOURCE
    assert "Opta-benzeri otomatik video analizi" in PANEL_SOURCE
    assert "Opta verisi" not in PANEL_SOURCE or "değildir" in PANEL_SOURCE


def test_panel_turkish_labels():
    frame = pd.DataFrame(
        [{"global_player_id": 1, "pass_attempts": 3, "pass_completion_pct": 66.7}]
    )
    renamed = rename_frame(frame, PLAYER_COLUMN_TR)
    assert "Pas denemesi" in renamed.columns
    assert "İsabetli pas (%)" in renamed.columns
    assert "pass_attempts" not in renamed.columns


def test_quality_warning():
    assert quality_label("low_metric_coverage", 0.2) in {"Düşük güven", "Mevcut değil", "Kısmi veri"}
    assert format_value(None) == "Mevcut değil"
    assert format_value(12.345, digits=1) == "12.3"
