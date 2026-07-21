"""Global identity resolution stage for Opta pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_analytics.opta.identity_resolve import (
    IdentityResolveConfig,
    build_track_fragments,
    resolve_global_identities,
)
from football_analytics.stages.base import Stage
from football_analytics.stages.mvp2_common import read_required_parquet
from football_analytics.utils.io import write_json


class GlobalIdentityStage(Stage):
    name = "global_identity"

    def validate_inputs(self) -> None:
        read_required_parquet(self.run_dir / "tracks.parquet")
        read_required_parquet(self.run_dir / "track_identities.parquet")

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        tracks = read_required_parquet(self.run_dir / "tracks.parquet")
        identities = read_required_parquet(self.run_dir / "track_identities.parquet")
        reid_path = self.run_dir / "track_reid_prototypes.parquet"
        reid = pd.read_parquet(reid_path) if reid_path.is_file() else None
        metrics_path = self.run_dir / "player_metrics.parquet"
        player_metrics = pd.read_parquet(metrics_path) if metrics_path.is_file() else None

        raw = self.config.get("global_identity") or {}
        cfg = IdentityResolveConfig(
            **{k: v for k, v in raw.items() if k in IdentityResolveConfig.__dataclass_fields__}
        )
        fragments = build_track_fragments(
            tracks, identities, reid, player_metrics, config=cfg
        )
        gmap, report, metrics, decisions = resolve_global_identities(fragments, config=cfg)

        gmap_path = self.run_dir / "global_identity_map.parquet"
        report_path = self.run_dir / "global_identity_report.parquet"
        gmap.to_parquet(gmap_path, index=False)
        report.to_parquet(report_path, index=False)
        if not decisions.empty:
            decisions.to_parquet(self.run_dir / "global_identity_decisions.parquet", index=False)
            # Explainable merge audit (required schema)
            audit_rows = []
            for _, d in decisions.iterrows():
                audit_rows.append(
                    {
                        "track_a": d.get("local_track_id"),
                        "track_b": d.get("candidate_global_id")
                        if pd.notna(d.get("candidate_global_id"))
                        else d.get("global_id"),
                        "decision": d.get("decision"),
                        "reid_similarity": d.get("reid_sim"),
                        "team_match": d.get("team_sim"),
                        "role_match": None
                        if d.get("reason") != "role_mismatch_gk_outfield"
                        else False,
                        "time_gap": d.get("gap_seconds"),
                        "pitch_distance": d.get("pos_delta"),
                        "simultaneous_overlap": d.get("reason") == "simultaneous_overlap",
                        "final_score": d.get("score"),
                        "rejection_reason": d.get("reason")
                        if str(d.get("decision")) == "reject"
                        else None,
                    }
                )
            pd.DataFrame(audit_rows).to_parquet(
                self.run_dir / "identity_merge_audit.parquet", index=False
            )

        write_json(self.run_dir / "identity_quality.json", metrics)
        write_json(self.stage_dir / "metrics.json", metrics)
        (self.run_dir / "stage_manifests").mkdir(parents=True, exist_ok=True)
        write_json(
            self.run_dir / "stage_manifests" / "global_identity.json",
            {"stage": self.name, "status": "PASS", **metrics},
        )
        return {
            "global_identity_map": gmap_path,
            "global_identity_report": report_path,
            "identity_quality": self.run_dir / "identity_quality.json",
            "metrics": self.stage_dir / "metrics.json",
        }

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        if not (self.run_dir / "global_identity_map.parquet").is_file():
            raise FileNotFoundError("global_identity_map.parquet")
        if not (self.run_dir / "identity_quality.json").is_file():
            raise FileNotFoundError("identity_quality.json")
