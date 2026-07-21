"""Action inference stage: passes, dribbles, duels, defensive actions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_analytics.opta.actions import (
    ActionInferenceConfig,
    infer_clearances,
    infer_dribbles,
    infer_duels_from_proximity,
    infer_passes,
    infer_turnovers_and_interceptions,
)
from football_analytics.opta.pitch_zones import PitchZones, infer_attack_sign_from_positions
from football_analytics.stages.base import Stage
from football_analytics.utils.io import write_json


class ActionInferenceStage(Stage):
    name = "action_inference"

    def validate_inputs(self) -> None:
        if not (self.run_dir / "touch_events.parquet").is_file():
            raise FileNotFoundError("touch_events.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        touches = pd.read_parquet(self.run_dir / "touch_events.parquet")
        # restore distance_m alias if renamed
        if "distance_to_ball" in touches.columns and "distance_m" not in touches.columns:
            touches = touches.rename(columns={"distance_to_ball": "distance_m"})
        game_path = self.run_dir / "game_state.parquet"
        game = pd.read_parquet(game_path) if game_path.is_file() else pd.DataFrame()
        id_path = self.run_dir / "track_identities.parquet"
        identities = pd.read_parquet(id_path) if id_path.is_file() else None

        team0_xs: list[float] = []
        team1_xs: list[float] = []
        if not game.empty and {"team_id", "x_field"} <= set(game.columns):
            for row in game.itertuples(index=False):
                if pd.isna(getattr(row, "x_field", None)):
                    continue
                tid = str(getattr(row, "team_id", ""))
                if tid.endswith("0"):
                    team0_xs.append(float(row.x_field))
                elif tid.endswith("1"):
                    team1_xs.append(float(row.x_field))
        zones = PitchZones(
            team0_attack_sign=infer_attack_sign_from_positions(team0_xs, team1_xs)
        )
        write_json(self.run_dir / "pitch_zones.json", zones.to_dict())

        raw = self.config.get("actions") or self.config.get("opta_actions") or {}
        cfg = ActionInferenceConfig(
            **{k: v for k, v in raw.items() if k in ActionInferenceConfig.__dataclass_fields__}
        )
        passes = infer_passes(touches, zones=zones, config=cfg)
        dribbles = infer_dribbles(touches, zones=zones, config=cfg)
        duels = infer_duels_from_proximity(
            game if not game.empty else pd.DataFrame(),
            touches,
            identities=identities,
            config=cfg,
        )
        # Aerial candidate only with airborne ball evidence
        ball_path = self.run_dir / "ball_state.parquet"
        if ball_path.is_file() and not touches.empty:
            ball = pd.read_parquet(ball_path)
            aerial_rows = []
            if "visibility_state" in ball.columns:
                airborne_frames = set(
                    ball.loc[ball["visibility_state"].astype(str).eq("airborne"), "frame_id"]
                    .astype(int)
                    .tolist()
                )
                for touch in touches.itertuples(index=False):
                    if int(getattr(touch, "frame_id", -1)) not in airborne_frames:
                        continue
                    aerial_rows.append(
                        {
                            "duel_id": f"aerial-cand-{getattr(touch, 'touch_id', 0)}",
                            "player_a": getattr(touch, "global_player_id", touch.track_id),
                            "player_b": None,
                            "team_a": getattr(touch, "team_id", None),
                            "team_b": None,
                            "timestamp_ms": float(touch.timestamp_ms),
                            "duel_type": "aerial_duel",
                            "winner_global_id": None,
                            "loser_global_id": None,
                            "confidence": 0.42,
                            "status": "candidate",
                        }
                    )
            if aerial_rows:
                aerial_df = pd.DataFrame(aerial_rows)
                duels = aerial_df if duels.empty else pd.concat([duels, aerial_df], ignore_index=True)

        clearances = infer_clearances(touches, zones=zones, config=cfg)
        defensive = infer_turnovers_and_interceptions(passes, config=cfg)
        if not clearances.empty:
            defensive = pd.concat([defensive, clearances], ignore_index=True)

        passes.to_parquet(self.run_dir / "pass_events.parquet", index=False)
        dribbles.to_parquet(self.run_dir / "dribble_events.parquet", index=False)
        duels.to_parquet(self.run_dir / "duel_events.parquet", index=False)
        defensive.to_parquet(self.run_dir / "defensive_actions.parquet", index=False)

        def _c(frame: pd.DataFrame, status: str) -> int:
            if frame is None or frame.empty or "status" not in frame.columns:
                return 0
            return int((frame["status"] == status).sum())

        metrics = {
            "pass_confirmed": _c(passes, "confirmed"),
            "pass_candidate": _c(passes, "candidate"),
            "dribble_confirmed": _c(dribbles, "confirmed"),
            "duel_confirmed": _c(duels, "confirmed"),
            "duel_unresolved": _c(duels, "unresolved"),
            "duel_candidate": _c(duels, "candidate"),
            "defensive_confirmed": _c(defensive, "confirmed"),
        }
        write_json(self.stage_dir / "metrics.json", metrics)
        (self.run_dir / "stage_manifests").mkdir(parents=True, exist_ok=True)
        write_json(
            self.run_dir / "stage_manifests" / "action_inference.json",
            {"stage": self.name, "status": "PASS", **metrics},
        )
        return {
            "pass_events": self.run_dir / "pass_events.parquet",
            "dribble_events": self.run_dir / "dribble_events.parquet",
            "duel_events": self.run_dir / "duel_events.parquet",
            "defensive_actions": self.run_dir / "defensive_actions.parquet",
            "pitch_zones": self.run_dir / "pitch_zones.json",
            "metrics": self.stage_dir / "metrics.json",
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        for name in (
            "pass_events.parquet",
            "dribble_events.parquet",
            "duel_events.parquet",
            "defensive_actions.parquet",
        ):
            if not (self.run_dir / name).is_file():
                raise FileNotFoundError(name)
