"""MP4 export for review clips (optional OpenCV path)."""

from __future__ import annotations

import pytest

from football_analytics.events import ClipWindow, EventType, export_clip_mp4, opencv_available

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

FPS = 25.0
WIDTH, HEIGHT = 64, 48


@pytest.fixture()
def source_video(tmp_path):
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    assert writer.isOpened()
    for i in range(100):
        frame = np.full((HEIGHT, WIDTH, 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestExportClipMp4:
    def test_reports_opencv_available(self):
        assert opencv_available() is True

    def test_writes_short_mp4_for_window(self, source_video, tmp_path):
        window = ClipWindow(
            event_id="goal-1",
            event_type=EventType.GOAL,
            start_ms=1_000.0,
            end_ms=2_000.0,
        )
        out = tmp_path / "clips" / "goal-1.mp4"
        result = export_clip_mp4(window, source_video, out)
        assert result["wrote_file"] is True
        # 1 second at 25 fps, inclusive frame range.
        assert result["frames_written"] == 26
        assert out.is_file() and out.stat().st_size > 0

        check = cv2.VideoCapture(str(out))
        assert check.isOpened()
        assert int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == 26
        check.release()

    def test_max_frames_caps_output(self, source_video, tmp_path):
        window = ClipWindow(
            event_id="goal-2",
            event_type=EventType.GOAL,
            start_ms=0.0,
            end_ms=60_000.0,
        )
        out = tmp_path / "goal-2.mp4"
        result = export_clip_mp4(window, source_video, out, max_frames=10)
        assert result["frames_written"] == 10

    def test_missing_source_raises(self, tmp_path):
        window = ClipWindow(
            event_id="goal-3",
            event_type=EventType.GOAL,
            start_ms=0.0,
            end_ms=1_000.0,
        )
        with pytest.raises(FileNotFoundError):
            export_clip_mp4(window, tmp_path / "missing.mp4", tmp_path / "out.mp4")
