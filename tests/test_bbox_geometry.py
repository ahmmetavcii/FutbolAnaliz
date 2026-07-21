import pytest

from football_analytics.geometry import (
    BBox,
    canonical_foot_point,
    clip_bbox,
    player_crop_quality,
)


def test_bbox_geometry_and_canonical_foot_point() -> None:
    box = BBox(10, 20, 30, 60)
    assert box.width == 20
    assert box.height == 40
    assert box.area == 800
    assert box.is_valid(min_area=800)
    assert canonical_foot_point(box) == (20, 60)


def test_clip_visibility_and_out_of_frame() -> None:
    box = BBox(-10, 10, 30, 50)
    assert clip_bbox(box, 100, 100) == BBox(0, 10, 30, 50)
    assert box.visible_fraction(100, 100) == pytest.approx(0.75)
    assert box.foot_point_confidence(100, 100) == pytest.approx(0.75)
    assert not box.is_out_of_frame(100, 100)
    assert BBox(110, 10, 130, 50).is_out_of_frame(100, 100)


def test_invalid_and_truncated_player_crop_quality() -> None:
    assert not BBox(5, 5, 5, 10).is_valid()
    assert not BBox(float("nan"), 0, 10, 10).is_valid()
    quality = player_crop_quality(
        BBox(10, 70, 30, 110),
        100,
        100,
        min_area=100,
        min_visible_fraction=0.5,
    )
    assert quality.valid
    assert quality.visible_fraction == pytest.approx(0.75)
    assert quality.foot_point_confidence == 0.0
    assert quality.score == 0.0


def test_bad_frame_size_rejected() -> None:
    with pytest.raises(ValueError):
        BBox(0, 0, 1, 1).clip(0, 100)
