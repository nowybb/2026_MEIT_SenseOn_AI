# tests/test_collision.py

import pytest

from ai2.collision import (
    calculate_bbox_zone_overlap,
    check_bbox_path_collision,
    check_path_collision,
)


ZONE = (40, 40, 60, 60)


# ============================================================
# 1. bbox 일부가 Zone과 겹치는 경우
# ============================================================

def test_partial_bbox_overlap():

    bbox = (25, 42, 45, 58)

    ratio = calculate_bbox_zone_overlap(
        bbox,
        ZONE
    )

    # 교집합 = 5 * 16
    # bbox 면적 = 20 * 16
    # 겹침 비율 = 0.25
    assert ratio == pytest.approx(0.25)


# ============================================================
# 2. bbox가 Zone 밖에 있는 경우
# ============================================================

def test_no_bbox_overlap():

    bbox = (0, 0, 10, 10)

    ratio = calculate_bbox_zone_overlap(
        bbox,
        ZONE
    )

    assert ratio == 0.0


# ============================================================
# 3. 중심점은 밖이지만 bbox 일부는 겹치는 경우
# ============================================================

def test_bbox_overlaps_even_if_center_outside():

    bbox = (25, 42, 45, 58)

    center = (35, 50)

    # 기존 중심점 방식: Zone 밖
    old_result = check_path_collision(
        center,
        center,
        ZONE
    )

    # 새로운 bbox 방식: 일부 겹침
    new_result = check_bbox_path_collision(
        bbox,
        center,
        center,
        ZONE
    )

    assert old_result is False
    assert new_result is True


# ============================================================
# 4. 이동 중 Zone을 통과하는 경우
# ============================================================

def test_bbox_path_crosses_zone():

    bbox = (0, 42, 10, 52)

    current = (5, 47)
    predicted = (95, 47)

    result = check_bbox_path_collision(
        bbox,
        current,
        predicted,
        ZONE
    )

    assert result is True


# ============================================================
# 5. Zone과 전혀 교차하지 않는 경우
# ============================================================

def test_bbox_path_misses_zone():

    bbox = (0, 0, 10, 10)

    current = (5, 5)
    predicted = (95, 5)

    result = check_bbox_path_collision(
        bbox,
        current,
        predicted,
        ZONE
    )

    assert result is False


# ============================================================
# 6. 빠르게 지나가도 교차를 놓치지 않는지 확인
# ============================================================

def test_fast_bbox_path_crosses_zone():

    bbox = (0, 42, 10, 52)

    current = (5, 47)
    predicted = (1000, 47)

    result = check_bbox_path_collision(
        bbox,
        current,
        predicted,
        ZONE
    )

    assert result is True


# ============================================================
# 7. 유효하지 않은 bbox 처리
# ============================================================

def test_invalid_bbox():

    result = check_bbox_path_collision(
        bbox=(10, 10, 10, 20),
        current_position=(10, 15),
        predicted_position=(50, 15),
        collision_zone=ZONE
    )

    assert result is False