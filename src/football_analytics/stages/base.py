"""Stage base contract."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from football_analytics.utils.hashing import sha256_file
from football_analytics.utils.io import write_json


class Stage(ABC):
    name: str

    def __init__(self, run_dir: Path, config: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.config = config
        self.stage_dir = run_dir / "stages" / self.name
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate_inputs(self) -> None: ...

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def run(self) -> dict[str, Any]: ...

    @abstractmethod
    def validate_outputs(self, artifacts: dict[str, Any]) -> None: ...

    def write_manifest(self, artifacts: dict[str, Any], status: str = "completed") -> Path:
        checksums: dict[str, str] = {}
        for key, value in artifacts.items():
            path = Path(value) if isinstance(value, (str, Path)) else None
            if path is not None and path.is_file():
                checksums[key] = sha256_file(path)
        payload = {
            "stage": self.name,
            "status": status,
            "completed_at": dt.datetime.now().astimezone().isoformat(),
            "schema_version": self.config.get("pipeline", {}).get(
                "schema_version", "1.0.0"
            ),
            "config_sha256": hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "artifacts": {k: str(v) for k, v in artifacts.items()},
            "checksums": checksums,
        }
        path = self.stage_dir / "stage_manifest.json"
        write_json(path, payload)
        return path

    def execute(self, mode: str = "auto") -> dict[str, Any]:
        """Run the stage.

        mode:
        - "auto": skip when a completed manifest matches the current config
          hash and all artifact checksums and output validation pass.
        - "trust": skip when checksums and output validation pass, ignoring
          the config hash (used for explicit --rerun-from resumes where
          earlier stages are known to be unaffected by the config change).
        - "force": always run.
        """
        if mode not in {"auto", "trust", "force"}:
            raise ValueError(f"unknown execute mode: {mode}")
        manifest_path = self.stage_dir / "stage_manifest.json"
        resume = bool(self.config.get("pipeline", {}).get("resume", False))
        if mode != "force" and resume and manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_config_hash = hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode("utf-8")
            ).hexdigest()
            config_ok = (
                mode == "trust"
                or payload.get("config_sha256") == current_config_hash
            )
            if payload.get("status") == "completed" and config_ok:
                artifacts = payload.get("artifacts", {})
                try:
                    for key, expected in payload.get("checksums", {}).items():
                        artifact = Path(artifacts[key])
                        if not artifact.is_file() or sha256_file(artifact) != expected:
                            raise RuntimeError(f"resume checksum mismatch: {key}")
                    self.validate_outputs(artifacts)
                except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                    pass
                else:
                    return artifacts
        self.validate_inputs()
        self.prepare()
        artifacts = self.run()
        self.validate_outputs(artifacts)
        self.write_manifest(artifacts, status="completed")
        return artifacts
