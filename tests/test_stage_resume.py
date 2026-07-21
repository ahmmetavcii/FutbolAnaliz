from __future__ import annotations

from pathlib import Path
from typing import Any

from football_analytics.stages.base import Stage


class CountingStage(Stage):
    name = "counting"

    def __init__(self, run_dir: Path, config: dict[str, Any]) -> None:
        super().__init__(run_dir, config)
        self.calls = 0

    def validate_inputs(self) -> None:
        return None

    def prepare(self) -> None:
        return None

    def run(self) -> dict[str, Any]:
        self.calls += 1
        artifact = self.run_dir / "count.txt"
        artifact.write_text("one\n")
        return {"count": artifact}

    def validate_outputs(self, artifacts: dict[str, Any]) -> None:
        if not Path(artifacts["count"]).is_file():
            raise RuntimeError("missing")


def test_completed_stage_is_validated_and_skipped(tmp_path: Path) -> None:
    config = {"pipeline": {"resume": True, "schema_version": "2.0.0"}}
    first = CountingStage(tmp_path, config)
    first.execute()
    assert first.calls == 1

    resumed = CountingStage(tmp_path, config)
    artifacts = resumed.execute()
    assert resumed.calls == 0
    assert Path(artifacts["count"]).read_text() == "one\n"
