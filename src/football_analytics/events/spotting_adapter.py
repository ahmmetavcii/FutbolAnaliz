"""SoccerNet action-spotting adapter (PTS-baseline / E2E-Spot).

Priority: official checkpoint inference via isolated ``sn-pts-baseline`` env.
Falls back to an empty candidate list when the worker is unavailable — never
treats an evaluator-only result as model inference.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PYTHON = Path("/home/ahmet/miniconda3/envs/sn-pts-baseline/bin/python")
DEFAULT_REPO = Path("/home/ahmet/projects/soccernet/PTS-baseline")
DEFAULT_CKPT = Path("/home/ahmet/models/soccernet/PTS-baseline")
DEFAULT_WORKER = Path(__file__).resolve().parents[3] / "scripts" / "pts_spotting_worker.py"


@dataclass
class SpottingCandidate:
    event_type: str
    timestamp: float
    confidence: float
    source_model: str
    temporal_window: tuple[int, int]
    frame_id: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpottingAdapterConfig:
    enabled: bool = True
    python: str = str(DEFAULT_PYTHON)
    repo_root: str = str(DEFAULT_REPO)
    checkpoint_dir: str = str(DEFAULT_CKPT)
    worker: str = str(DEFAULT_WORKER)
    device: str = "cuda"
    score_threshold: float = 0.35
    max_frames: int = 100
    timeout_seconds: float = 600.0


class SpottingAdapter:
    """Adapter contract for event timestamp candidates."""

    def __init__(self, config: SpottingAdapterConfig | None = None) -> None:
        self.config = config or SpottingAdapterConfig()
        self._loaded = False
        self._last_payload: dict[str, Any] = {}

    def validate_inputs(self, video: Path) -> None:
        if not Path(video).is_file():
            raise FileNotFoundError(video)
        if self.config.enabled:
            if not Path(self.config.python).is_file():
                raise FileNotFoundError(f"spotting python missing: {self.config.python}")
            if not Path(self.config.checkpoint_dir, "checkpoint_088.pt").is_file():
                raise FileNotFoundError("PTS-baseline checkpoint_088.pt missing")

    def load_model(self) -> None:
        # Model loads inside the worker process.
        self._loaded = True

    def extract_features(self, video: Path) -> dict[str, Any]:
        return {"video": str(video)}

    def predict_events(self, video: Path, output_json: Path) -> list[SpottingCandidate]:
        if not self.config.enabled:
            self._last_payload = {"status": "DISABLED", "candidates": []}
            return []
        self.validate_inputs(video)
        self.load_model()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.python,
            self.config.worker,
            "--repo-root",
            self.config.repo_root,
            "--checkpoint-dir",
            self.config.checkpoint_dir,
            "--video",
            str(video),
            "--output",
            str(output_json),
            "--device",
            self.config.device,
            "--max-frames",
            str(self.config.max_frames),
            "--score-threshold",
            str(self.config.score_threshold),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        if result.returncode != 0 or not output_json.is_file():
            self._last_payload = {
                "status": "WORKER_FAILED",
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[-2000:],
                "candidates": [],
            }
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(self._last_payload, indent=2), encoding="utf-8"
            )
            return []
        return self.decode_predictions(output_json)

    def decode_predictions(self, path: Path) -> list[SpottingCandidate]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._last_payload = payload
        candidates: list[SpottingCandidate] = []
        for row in payload.get("candidates", []):
            window = row.get("temporal_window") or [row.get("frame_id", 0), row.get("frame_id", 0)]
            candidates.append(
                SpottingCandidate(
                    event_type=str(row["event_type"]),
                    timestamp=float(row["timestamp"]),
                    confidence=float(row["confidence"]),
                    source_model=str(row.get("source_model", "PTS-baseline/E2E-Spot")),
                    temporal_window=(int(window[0]), int(window[1])),
                    frame_id=int(row["frame_id"]) if row.get("frame_id") is not None else None,
                    raw=dict(row),
                )
            )
        return candidates

    def validate_outputs(self, candidates: list[SpottingCandidate]) -> None:
        for item in candidates:
            if not 0.0 <= item.confidence <= 1.0:
                raise ValueError(f"invalid spotting confidence: {item.confidence}")

    def cleanup(self) -> None:
        self._loaded = False

    @property
    def last_payload(self) -> dict[str, Any]:
        return dict(self._last_payload)
