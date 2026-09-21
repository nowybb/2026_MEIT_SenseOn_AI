import math

from ai2.ttc import (
    calculate_visual_ttc,
    calculate_visual_ttc_robust,
)


def make_detection(
    timestamp,
    width,
    height,
):
    return {
        "timestamp": timestamp,
        "bbox": [
            0,
            0,
            width,
            height,
        ],
    }


def test_old_ttc_approaching():

    history = [
        make_detection(0.0, 100, 100),
        make_detection(0.1, 110, 110),
        make_detection(0.2, 120, 120),
        make_detection(0.3, 130, 130),
        make_detection(0.4, 140, 140),
    ]

    ttc = calculate_visual_ttc(
        history
    )

    assert ttc is not None
    assert ttc > 0


def test_robust_ttc_approaching():

    history = [
        make_detection(0.0, 100, 100),
        make_detection(0.1, 110, 110),
        make_detection(0.2, 120, 120),
        make_detection(0.3, 130, 130),
        make_detection(0.4, 140, 140),
    ]

    ttc = calculate_visual_ttc_robust(
        history
    )

    assert ttc is not None
    assert ttc > 0


def test_robust_ttc_stationary():

    history = [
        make_detection(0.0, 100, 100),
        make_detection(0.1, 100, 100),
        make_detection(0.2, 100, 100),
        make_detection(0.3, 100, 100),
        make_detection(0.4, 100, 100),
    ]

    ttc = calculate_visual_ttc_robust(
        history
    )

    assert ttc is None


def test_robust_ttc_receding():

    history = [
        make_detection(0.0, 140, 140),
        make_detection(0.1, 130, 130),
        make_detection(0.2, 120, 120),
        make_detection(0.3, 110, 110),
        make_detection(0.4, 100, 100),
    ]

    ttc = calculate_visual_ttc_robust(
        history
    )

    assert ttc is None


def test_robust_ttc_handles_bbox_spike():

    history = [
        make_detection(0.0, 100, 100),
        make_detection(0.1, 110, 110),
        make_detection(0.2, 250, 250),  # 이상치
        make_detection(0.3, 130, 130),
        make_detection(0.4, 140, 140),
        make_detection(0.5, 150, 150),
    ]

    ttc = calculate_visual_ttc_robust(
        history
    )

    assert ttc is not None

    # 정상적인 증가 추세가 유지되어야 한다.
    assert math.isfinite(ttc)
    assert ttc > 0


def test_robust_ttc_receding_with_bbox_spike():

    history = [
        make_detection(0.0, 150, 150),
        make_detection(0.1, 140, 140),
        make_detection(0.2, 220, 220),  # 이상치
        make_detection(0.3, 120, 120),
        make_detection(0.4, 110, 110),
        make_detection(0.5, 100, 100),
    ]

    ttc = calculate_visual_ttc_robust(
        history
    )

    assert ttc is None


def test_robust_ttc_insufficient_history():

    history = [
        make_detection(
            0.0,
            100,
            100,
        ),
        make_detection(
            0.1,
            110,
            110,
        ),
    ]

    ttc = calculate_visual_ttc_robust(
        history
    )

    assert ttc is None