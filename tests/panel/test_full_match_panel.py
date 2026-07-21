from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "apps" / "full_match_panel.py"


def load_panel():
    spec = importlib.util.spec_from_file_location("full_match_panel_under_test", PANEL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panel_import_does_not_import_streamlit(monkeypatch):
    sys.modules.pop("streamlit", None)
    original = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "streamlit":
            raise AssertionError("streamlit was imported while loading the panel")
        return original(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    panel = load_panel()

    assert panel.SECTION_NAMES[0] == "New Match"
    assert panel.SECTION_NAMES[-1] == "Results"


def test_path_safety_rejects_traversal_and_prefix_confusion(tmp_path):
    panel = load_panel()
    root = tmp_path / "uploads"

    assert panel.safe_path_under(root, "match/video.mp4") == (
        root / "match" / "video.mp4"
    ).resolve()
    with pytest.raises(ValueError):
        panel.safe_path_under(root, "../outside.mp4")
    with pytest.raises(ValueError):
        panel.safe_path_under(root, tmp_path / "uploads-elsewhere" / "video.mp4")


@pytest.mark.parametrize("camera_count", [1, 2, 4])
def test_manifest_payload_supports_required_camera_counts(camera_count):
    panel = load_panel()
    cameras = [
        panel.UPLOAD_ROOT / "match-01" / f"camera_{index}.mp4"
        for index in range(1, camera_count + 1)
    ]

    payload = panel.build_manifest_payload("match-01", cameras)

    assert payload["schema_version"] == "1.0.0"
    assert payload["match_id"] == "match-01"
    assert payload["camera_count"] == camera_count
    assert [item["camera_id"] for item in payload["cameras"]] == [
        f"camera_{index}" for index in range(1, camera_count + 1)
    ]
    assert all(item["offset_seconds"] == 0.0 for item in payload["cameras"])


def test_manifest_payload_rejects_unsupported_camera_count():
    panel = load_panel()
    cameras = [panel.UPLOAD_ROOT / f"camera_{index}.mp4" for index in range(3)]

    with pytest.raises(ValueError, match="1, 2, or 4"):
        panel.build_manifest_payload("match-01", cameras)


def test_combined_correction_payload_is_normalized():
    panel = load_panel()

    payload = panel.build_correction_payload(
        "match-01",
        merges=[("old-7", "player-7")],
        splits=[
            {
                "identity": "player-9",
                "track_ids": [41, 42],
                "new_identity": "player-10",
            }
        ],
        roles=[{"global_identity": "player-7", "role": "goalkeeper"}],
        events=[{"type": "goal", "timestamp": 93.5, "team": "home"}],
    )

    assert payload["identity_corrections"]["merges"] == [
        {"source_identity": "old-7", "target_identity": "player-7"}
    ]
    assert payload["identity_corrections"]["splits"][0]["track_ids"] == ["41", "42"]
    assert payload["role_corrections"] == [
        {"identity": "player-7", "role": "goalkeeper"}
    ]
    assert payload["event_corrections"] == [
        {"event_type": "goal", "timestamp_seconds": 93.5, "team": "home"}
    ]


def test_run_command_is_argument_list_and_confines_paths():
    panel = load_panel()
    video = panel.UPLOAD_ROOT / "match-01" / "camera_1.mp4"
    destination = panel.RESULTS_ROOT / "match-01"

    command = panel.build_run_command(video, destination, python_executable="python")

    assert command[0] == "python"
    assert command[command.index("--input") + 1] == str(video)
    assert command[command.index("--runs-root") + 1] == str(destination)
    with pytest.raises(ValueError):
        panel.build_run_command("/tmp/untrusted.mp4", destination)
