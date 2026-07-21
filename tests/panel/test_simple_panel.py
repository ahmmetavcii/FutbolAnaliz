"""Tests for the simplified single-button panel and its analysis driver."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PROJECT_ROOT / "apps" / "full_match_panel.py"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

spec = importlib.util.spec_from_file_location("full_match_panel", PANEL_PATH)
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)

from football_analytics.full_match import panel_driver as driver  # noqa: E402

PANEL_SOURCE = PANEL_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Simple mode defaults and UI structure.
# ---------------------------------------------------------------------------


def test_simple_mode_is_default():
    assert panel.SIMPLE_MODE_DEFAULT is True


def test_upload_is_visible_on_main_screen():
    assert "file_uploader" in PANEL_SOURCE
    assert "MP4, MOV, MKV" in PANEL_SOURCE


def test_single_start_button():
    assert PANEL_SOURCE.count('"ANALİZİ BAŞLAT"') == 1


def test_technical_details_hidden_by_default():
    assert panel.TECHNICAL_DETAILS_HIDDEN_DEFAULT is True
    assert '"Teknik ayrıntıları göster", expanded=False' in PANEL_SOURCE
    assert '"Gelişmiş Ayarlar", expanded=False' in PANEL_SOURCE


def test_no_separate_prepare_export_resume_buttons_in_simple_flow():
    # The simple render path exposes exactly one primary action; the legacy
    # operator panel is retained only as an unused helper.
    for label in ("Prepare and start", "Resume run"):
        assert label not in PANEL_SOURCE.split("def _render_legacy_panel")[0]


def test_video_player_and_excel_download_present():
    assert "st.video(str(annotated))" in PANEL_SOURCE
    assert "full_match_report.xlsx" in PANEL_SOURCE
    assert "download_button" in PANEL_SOURCE


def test_automatic_refresh_configured():
    assert 2.0 <= panel.AUTO_REFRESH_SECONDS <= 3.0
    running_section = PANEL_SOURCE.split('if status == "RUNNING":')[1]
    assert "st.rerun()" in running_section


# ---------------------------------------------------------------------------
# One-button flow wiring.
# ---------------------------------------------------------------------------


def test_panel_analysis_command_targets_single_flow_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(panel, "RESULTS_ROOT", tmp_path / "results")
    video = tmp_path / "uploads" / "m1" / "camera_1_a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    command = panel.build_panel_analysis_command(video, tmp_path / "results" / "m1")
    assert command[1].endswith("run_panel_analysis.py")
    assert "--video" in command and "--match-dir" in command
    resumed = panel.build_panel_analysis_command(
        video, tmp_path / "results" / "m1", resume=True
    )
    assert "--resume" in resumed


def test_start_analysis_spawns_driver_once(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(panel, "RESULTS_ROOT", tmp_path / "results")
    (tmp_path / "results").mkdir()
    video = tmp_path / "uploads" / "m1" / "camera_1_a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    calls: list[list[str]] = []

    class FakeProcess:
        pid = 4242

    def fake_popen(command, **kwargs):
        calls.append(list(command))
        return FakeProcess()

    monkeypatch.setattr(panel.subprocess, "Popen", fake_popen)
    pointer = panel.start_panel_analysis(video, "m1")
    assert pointer["pid"] == 4242
    assert len(calls) == 1
    assert calls[0][1].endswith("run_panel_analysis.py")
    saved = driver.load_active_pointer(tmp_path / "results")
    assert saved["match_id"] == "m1"


def test_duplicate_process_prevention(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(panel, "RESULTS_ROOT", tmp_path / "results")
    results = tmp_path / "results"
    match_dir = results / "m1"
    match_dir.mkdir(parents=True)
    driver.write_state(match_dir, status="RUNNING", pid=os.getpid())
    driver.save_active_pointer(
        results, {"match_id": "m1", "match_dir": str(match_dir), "pid": os.getpid()}
    )
    video = tmp_path / "uploads" / "m2" / "camera_1_b.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Devam eden"):
        panel.start_panel_analysis(video, "m2")


# ---------------------------------------------------------------------------
# Progress mapping from real artifacts.
# ---------------------------------------------------------------------------


def test_progress_stage_weights_match_spec():
    assert driver.stage_percent("preparing") == 0.0
    assert driver.stage_percent("detection") == 10.0
    assert driver.stage_percent("tracking") == 32.0
    assert driver.stage_percent("tracking", 0.5) == 41.0
    assert driver.stage_percent("calibration") == 50.0
    assert driver.stage_percent("identity") == 58.0
    assert driver.stage_percent("jersey") == 66.0
    assert driver.stage_percent("metrics") == 72.0
    assert driver.stage_percent("events") == 78.0
    assert driver.stage_percent("render") == 88.0
    assert driver.stage_percent("reports", 1.0) == 100.0


def test_compute_progress_reads_pipeline_stage_manifests(tmp_path):
    match_dir = tmp_path / "m1"
    run = match_dir / "chunk_artifacts" / "camera_1" / "chunk_00000" / "pipeline_runs" / "run_x"
    for stage in ("ingest", "shot_classification", "detection"):
        stage_dir = run / "stages" / stage
        stage_dir.mkdir(parents=True)
        (stage_dir / "stage_manifest.json").write_text("{}")
    driver.write_state(match_dir, status="RUNNING", phase="pipeline", total_frames=750)
    progress = driver.compute_progress(match_dir)
    # detection done, tracking manifest missing -> currently tracking.
    assert progress["stage_key"] == "tracking"
    assert progress["percent"] == 32.0
    assert progress["total_frames"] == 750


def test_compute_progress_completed_is_100(tmp_path):
    match_dir = tmp_path / "m1"
    driver.write_state(match_dir, status="COMPLETED", phase="reports")
    progress = driver.compute_progress(match_dir)
    assert progress["percent"] == 100.0


# ---------------------------------------------------------------------------
# Heartbeat freshness and stall detection.
# ---------------------------------------------------------------------------


def test_heartbeat_fresh_and_stale_classification():
    assert driver.classify_heartbeat_age(5) == "Çalışıyor"
    assert driver.classify_heartbeat_age(30) == "Çalışıyor"
    assert driver.classify_heartbeat_age(45) == "Yavaş çalışıyor"
    assert driver.classify_heartbeat_age(120) == "Yanıt bekleniyor"
    assert driver.classify_heartbeat_age(181) == "İşlem takılmış olabilir"
    assert driver.classify_heartbeat_age(None) == "Bilgi yok"


def test_heartbeat_writer_produces_required_fields(tmp_path):
    match_dir = tmp_path / "m1"
    driver.write_state(match_dir, status="RUNNING", phase="validating", total_frames=10)
    writer = driver.HeartbeatWriter(match_dir)
    payload = writer.write_once()
    for field in (
        "timestamp", "pid", "current_stage", "current_camera", "current_chunk",
        "processed_frames", "total_frames", "last_log_line", "status",
    ):
        assert field in payload
    on_disk = driver.read_heartbeat(match_dir)
    assert on_disk["pid"] == os.getpid()
    age = driver.heartbeat_age_seconds(on_disk)
    assert age is not None and age < 5


# ---------------------------------------------------------------------------
# PID lifecycle: dead, zombie, cleanup.
# ---------------------------------------------------------------------------


def test_dead_pid_detected():
    assert driver.pid_status(2**22 + 12345) == "dead"
    assert driver.pid_status(None) == "dead"


def test_zombie_pid_not_reported_running():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    time.sleep(0.6)  # child exits and becomes a zombie until reaped
    status = driver.pid_status(process.pid)
    assert status in {"zombie", "dead"}
    assert status != "running"
    driver.reap_children()
    process.poll()
    assert driver.pid_status(process.pid) == "dead"


# ---------------------------------------------------------------------------
# Session persistence and recovery.
# ---------------------------------------------------------------------------


def test_active_pointer_roundtrip(tmp_path):
    driver.save_active_pointer(tmp_path, {"match_id": "m1", "pid": 7})
    assert driver.load_active_pointer(tmp_path)["match_id"] == "m1"


def test_load_current_analysis_recovers_after_refresh(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "RESULTS_ROOT", tmp_path)
    match_dir = tmp_path / "m1"
    match_dir.mkdir()
    driver.write_state(match_dir, status="COMPLETED", run_id="m1")
    driver.save_active_pointer(
        tmp_path, {"match_id": "m1", "match_dir": str(match_dir), "pid": 1}
    )
    current = panel.load_current_analysis()
    assert current["status"] == "COMPLETED"
    assert current["state"]["run_id"] == "m1"


def test_running_state_with_dead_pid_becomes_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "RESULTS_ROOT", tmp_path)
    match_dir = tmp_path / "m1"
    match_dir.mkdir()
    driver.write_state(match_dir, status="RUNNING", run_id="m1")
    driver.save_active_pointer(
        tmp_path,
        {"match_id": "m1", "match_dir": str(match_dir), "pid": 2**22 + 4321},
    )
    current = panel.load_current_analysis()
    assert current["status"] == "FAILED"


# ---------------------------------------------------------------------------
# Resume visibility rules.
# ---------------------------------------------------------------------------


def test_resume_hidden_while_healthy():
    state = {"status": "RUNNING", "pid": os.getpid(), "run_id": "m1"}
    assert driver.should_offer_resume(state, heartbeat_age=5) is False


def test_resume_offered_after_failure_or_stall():
    assert driver.should_offer_resume({"status": "FAILED", "run_id": "m1"}, None) is True
    dead = {"status": "RUNNING", "pid": 2**22 + 999, "run_id": "m1"}
    assert driver.should_offer_resume(dead, heartbeat_age=10) is True
    stale = {"status": "RUNNING", "pid": os.getpid(), "run_id": "m1"}
    assert driver.should_offer_resume(stale, heartbeat_age=400) is True


# ---------------------------------------------------------------------------
# Results display and failure messaging.
# ---------------------------------------------------------------------------


def test_results_display_uses_not_available_not_zero():
    assert panel._display_value(None) == "Mevcut değil"
    assert panel._display_value("") == "Mevcut değil"
    assert panel._display_value(float("nan")) == "Mevcut değil"
    assert panel._display_value(3.14159) == "3.14"


def test_failure_view_has_no_raw_traceback():
    state = {
        "status": "FAILED",
        "failed_stage": "Oyuncular tespit ediliyor",
        "error": "RuntimeError: kısa açıklama",
        "traceback": "Traceback (most recent call last): ...",
        "run_id": "m1",
    }
    view = panel.build_failure_view(state, "last line")
    assert view["title"] == "Analiz tamamlanamadı"
    assert "Traceback" not in json.dumps({k: v for k, v in view.items()})
    assert view["failed_stage"] == "Oyuncular tespit ediliyor"
    assert view["can_resume"] == "Evet"
    assert view["last_log_line"] == "last line"


def test_archive_instead_of_delete(tmp_path):
    match_dir = tmp_path / "m1"
    (match_dir / "run").mkdir(parents=True)
    (match_dir / "run" / "run_state.json").write_text("{}")
    driver.write_state(match_dir, status="FAILED")
    archive = driver.archive_previous_run(match_dir)
    assert archive is not None and archive.is_dir()
    assert (archive / "run" / "run_state.json").is_file()
    assert not (match_dir / "run").exists()


# ---------------------------------------------------------------------------
# Misc helpers.
# ---------------------------------------------------------------------------


def test_rendered_panel_is_simple(tmp_path, monkeypatch):
    """Render the real app: title + no technical widgets on the main screen."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PANEL_PATH), default_timeout=60)
    app.run()
    assert not app.exception
    assert [t.value for t in app.title] == ["Football Match Analysis"]
    labels = [e.label for e in app.expander]
    assert "Gelişmiş Ayarlar" in labels
    # None of the technical sections of the legacy operator panel render.
    for hidden in ("New Match", "Process Management", "Synchronization", "Calibration"):
        assert hidden not in labels
    rendered = " ".join(str(getattr(item, "value", "")) for item in app.markdown)
    assert "PID" not in rendered
    assert "manifest" not in rendered.lower()


def test_match_name_derived_from_filename():
    assert panel.derive_match_name("My Final (2026).mp4") == "My-Final-2026"
    assert panel.derive_match_name("") == "match"


def test_eta_from_real_percent_only():
    assert panel.estimate_remaining_seconds(0.0, 100.0) is None
    assert panel.estimate_remaining_seconds(50.0, 60.0) == 60.0
    assert panel.estimate_remaining_seconds(100.0, 60.0) == 0.0
