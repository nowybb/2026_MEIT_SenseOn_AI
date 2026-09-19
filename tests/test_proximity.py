import pytest

from ai2.proximity import calculate_proximity_features


def test_proximity_features():

    result = calculate_proximity_features(
        bbox=[400, 300, 500, 400],
        frame_width=1000,
        frame_height=500
    )

    assert result["height_ratio"] == pytest.approx(0.2)

    assert result["area_ratio"] == pytest.approx(0.02)

    assert result["bottom_y_ratio"] == pytest.approx(0.8)


def test_larger_bbox_has_larger_ratios():

    small = calculate_proximity_features(
        bbox=[400, 300, 450, 350],
        frame_width=1000,
        frame_height=500
    )

    large = calculate_proximity_features(
        bbox=[400, 250, 600, 450],
        frame_width=1000,
        frame_height=500
    )

    assert (
        large["height_ratio"]
        > small["height_ratio"]
    )

    assert (
        large["area_ratio"]
        > small["area_ratio"]
    )


def test_invalid_frame_size():

    with pytest.raises(ValueError):

        calculate_proximity_features(
            bbox=[0, 0, 100, 100],
            frame_width=0,
            frame_height=500
        )