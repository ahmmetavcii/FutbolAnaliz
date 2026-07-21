"""Turkish labels and formatting helpers for Opta-like panel tabs."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

_NOT_AVAILABLE = "Mevcut değil"
_PARTIAL = "Kısmi veri"
_LOW_CONF = "Düşük güven"

PLAYER_COLUMN_TR = {
    "global_player_id": "Küresel oyuncu ID",
    "team_id": "Takım",
    "jersey_number": "Forma no",
    "role": "Rol",
    "visible_seconds": "Görünür süre (sn)",
    "identity_quality": "Kimlik kalitesi",
    "metric_quality": "Metrik kalitesi",
    "distance_m": "Mesafe (m)",
    "max_speed_kmh": "Maks. hız (km/s)",
    "sprint_count": "Sprint sayısı",
    "pass_attempts": "Pas denemesi",
    "passes_completed": "Başarılı pas",
    "pass_completion_pct": "İsabetli pas (%)",
    "long_pass_attempts": "Uzun pas denemesi",
    "long_passes_completed": "Başarılı uzun pas",
    "long_pass_completion_pct": "Uzun pas başarı (%)",
    "zone_1_to_2_passes": "1→2 bölge pası",
    "zone_2_to_3_passes": "2→3 bölge pası",
    "dribble_attempts": "Dribbling denemesi",
    "dribbles_completed": "Başarılı dribbling",
    "dribble_success_pct": "Adam eksiltme (%)",
    "duels": "İkili mücadele",
    "duels_won": "Kazanılan mücadele",
    "duel_win_pct": "Mücadele kazanma (%)",
    "aerial_duels": "Hava topu mücadelesi",
    "aerial_duels_won": "Kazanılan hava topu",
    "tackles_won": "Top çalma",
    "interceptions": "Pas kesme",
    "clearances": "Uzaklaştırma",
    "turnovers": "Top kaybı",
    "dispossessions": "Top bırakma",
    "penalty_area_touches": "Ceza sahası teması",
    "activity_index": "Aktivasyon skoru",
    "quality_flags": "Kalite bayrakları",
}

TEAM_COLUMN_TR = {
    "team_id": "Takım",
    "validated_player_count": "Doğrulanmış oyuncu",
    "metric_coverage": "Metrik kapsama",
    "total_distance_m": "Toplam mesafe (m)",
    "sprint_count": "Sprint sayısı",
    "pass_attempts": "Pas denemesi",
    "passes_completed": "Başarılı pas",
    "pass_completion_pct": "İsabetli pas (%)",
    "long_pass_attempts": "Uzun pas denemesi",
    "long_passes_completed": "Başarılı uzun pas",
    "long_pass_completion_pct": "Uzun pas başarı (%)",
    "zone_1_to_2_passes": "1→2 bölge pası",
    "zone_2_to_3_passes": "2→3 bölge pası",
    "dribble_attempts": "Dribbling denemesi",
    "dribbles_completed": "Başarılı dribbling",
    "dribble_success_pct": "Adam eksiltme (%)",
    "duels": "İkili mücadele",
    "duels_won": "Kazanılan mücadele",
    "duel_win_pct": "Mücadele kazanma (%)",
    "aerial_duels": "Hava topu mücadelesi",
    "aerial_duels_won": "Kazanılan hava topu",
    "tackles_won": "Top çalma",
    "interceptions": "Pas kesme",
    "clearances": "Uzaklaştırma",
    "turnovers": "Top kaybı",
    "penalty_area_touches": "Ceza sahası teması",
}

PASS_COLUMN_TR = {
    "pass_id": "Pas ID",
    "passer_global_id": "Pas veren",
    "receiver_global_id": "Pas alan",
    "team_id": "Takım",
    "start_time_ms": "Başlangıç (ms)",
    "end_time_ms": "Bitiş (ms)",
    "start_zone": "Başlangıç bölgesi",
    "end_zone": "Bitiş bölgesi",
    "distance_m": "Mesafe (m)",
    "forward_progress_m": "İleri ilerleme (m)",
    "successful": "Başarılı",
    "long_pass": "Uzun pas",
    "progressive_pass": "Progresif pas",
    "confidence": "Güven",
    "status": "Durum",
}

DUEL_COLUMN_TR = {
    "duel_id": "Mücadele ID",
    "player_a": "Oyuncu A",
    "player_b": "Oyuncu B",
    "team_a": "Takım A",
    "team_b": "Takım B",
    "timestamp_ms": "Zaman (ms)",
    "duel_type": "Tür",
    "winner_global_id": "Kazanan",
    "loser_global_id": "Kaybeden",
    "confidence": "Güven",
    "status": "Durum",
}

DRIBBLE_COLUMN_TR = {
    "dribble_id": "Dribbling ID",
    "attacker_global_id": "Hücum oyuncusu",
    "defender_global_id": "Savunma oyuncusu",
    "timestamp_ms": "Zaman (ms)",
    "distance_m": "Mesafe (m)",
    "successful": "Başarılı",
    "defender_beaten": "Rakip geçildi",
    "confidence": "Güven",
    "status": "Durum",
}

DEFENSIVE_COLUMN_TR = {
    "action_id": "Aksiyon ID",
    "action_type": "Tür",
    "global_player_id": "Oyuncu",
    "team_id": "Takım",
    "timestamp_ms": "Zaman (ms)",
    "start_zone": "Başlangıç bölgesi",
    "end_zone": "Bitiş bölgesi",
    "distance_m": "Mesafe (m)",
    "under_pressure": "Baskı altında",
    "confidence": "Güven",
    "status": "Durum",
}

PHYSICAL_COLUMN_TR = {
    "track_id": "Takip ID",
    "timestamp_ms": "Zaman (ms)",
    "cumulative_distance_m": "Kümülatif mesafe (m)",
    "smoothed_speed_kmh": "Hız (km/s)",
    "sprint_state": "Sprint durumu",
    "valid": "Geçerli",
}

TAB_NAMES = [
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
]


def quality_label(flags: str | None, metric_quality: float | None = None) -> str:
    text = str(flags or "")
    if "insufficient" in text or metric_quality is not None and metric_quality < 0.2:
        return _NOT_AVAILABLE
    if "low_" in text or (metric_quality is not None and metric_quality < 0.45):
        return _LOW_CONF
    if "partial" in text or (metric_quality is not None and metric_quality < 0.7):
        return _PARTIAL
    return "Tam"


def format_value(value: Any, *, pct: bool = False, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return _NOT_AVAILABLE
    if pct:
        return f"{float(value):.{digits}f}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def rename_frame(frame: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    cols = {c: mapping[c] for c in frame.columns if c in mapping}
    out = frame.rename(columns=cols)
    # Drop unmapped English technical columns from display when mapping exists
    keep = [mapping[c] for c in frame.columns if c in mapping]
    return out[keep] if keep else out


def load_csv_or_empty(path) -> pd.DataFrame:
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_parquet_or_empty(path) -> pd.DataFrame:
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_parquet(p)


def status_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame is None or frame.empty or "status" not in frame.columns:
        return {"confirmed_count": 0, "candidate_count": 0, "unresolved_count": 0}
    return {
        "confirmed_count": int((frame["status"] == "confirmed").sum()),
        "candidate_count": int((frame["status"] == "candidate").sum()),
        "unresolved_count": int((frame["status"] == "unresolved").sum()),
    }
