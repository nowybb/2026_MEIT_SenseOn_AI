from ai2.approach import (
    calculate_approach_rate,
    is_approaching,
)


def make_detection(timestamp, bbox):
    return {
        "timestamp": timestamp,
        "bbox": bbox,
    }


def make_centered_bbox(width, height):
    """
    중심 위치는 동일하게 유지하고
    bbox 크기만 변경한다.
    """
    cx = 320
    cy = 240

    return [
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    ]


# ============================================================
# 1. 지속적으로 접근하는 객체
# ============================================================

def test_approaching_object():

    sizes = [
        (60, 90),
        (65, 98),
        (70, 105),
        (76, 114),
        (82, 123),
        (90, 135),
    ]

    history = [
        make_detection(
            i * 0.1,
            make_centered_bbox(width, height)
        )
        for i, (width, height) in enumerate(sizes)
    ]

    rate = calculate_approach_rate(history)

    assert rate > 0
    assert is_approaching(history) is True


# ============================================================
# 2. 정지 객체
# ============================================================

def test_stationary_object():

    history = [
        make_detection(
            i * 0.1,
            make_centered_bbox(80, 120)
        )
        for i in range(6)
    ]

    rate = calculate_approach_rate(history)

    assert abs(rate) < 1e-9
    assert is_approaching(history) is False


# ============================================================
# 3. 멀어지는 객체
# ============================================================

def test_receding_object():

    sizes = [
        (100, 150),
        (94, 141),
        (88, 132),
        (82, 123),
        (76, 114),
        (70, 105),
    ]

    history = [
        make_detection(
            i * 0.1,
            make_centered_bbox(width, height)
        )
        for i, (width, height) in enumerate(sizes)
    ]

    rate = calculate_approach_rate(history)

    assert rate < 0
    assert is_approaching(history) is False


# ============================================================
# 4. 정지 객체 + bbox jitter
# ============================================================

def test_stationary_bbox_jitter():

    sizes = [
        (80, 120),
        (82, 121),
        (79, 119),
        (81, 122),
        (80, 118),
        (81, 120),
        (79, 121),
        (80, 120),
    ]

    history = [
        make_detection(
            i * 0.1,
            make_centered_bbox(width, height)
        )
        for i, (width, height) in enumerate(sizes)
    ]

    assert is_approaching(history) is False


# ============================================================
# 5. 멀어지는 중 순간적인 bbox 증가
# ============================================================

def test_receding_with_bbox_spike():

    sizes = [
        (110, 165),
        (104, 156),
        (98, 147),

        # 순간적인 bbox 증가
        (103, 154),

        (90, 135),
        (85, 128),

        # 또 한 번의 작은 흔들림
        (88, 132),

        (78, 117),
        (73, 110),
        (68, 102),
    ]

    history = [
        make_detection(
            i * 0.1,
            make_centered_bbox(width, height)
        )
        for i, (width, height) in enumerate(sizes)
    ]

    rate = calculate_approach_rate(history)

    assert rate < 0
    assert is_approaching(history) is False