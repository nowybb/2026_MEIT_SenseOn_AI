# tests/test_collision.py

import pytest

from ai2.collision import (
    create_collision_zone,
    is_point_in_zone,
    check_path_collision,
    calculate_bbox_zone_overlap_ratio,
    predict_future_bbox,
    check_bbox_path_collision,
    calculate_bbox_zone_overlap,
    check_bbox_path_collision_exact,
)


# ============================================================
# 공통 Collision Zone
# main 테스트에서 사용
# ============================================================

ZONE = (
    40,
    40,
    60,
    60,
)


# ============================================================
# 1. 기존 Collision Zone 테스트
# ============================================================

def test_create_collision_zone():

    zone = create_collision_zone(
        1000,
        1000,
    )

    assert zone == (
        350.0,
        550.0,
        650.0,
        1000.0,
    )


def test_point_inside_collision_zone():

    zone = (
        350,
        550,
        650,
        1000,
    )

    assert is_point_in_zone(
        500,
        700,
        zone,
    )


# ============================================================
# 2. 기존 center 기반 collision 테스트
# ============================================================

def test_center_path_collision():

    zone = (
        350,
        550,
        650,
        1000,
    )

    result = check_path_collision(
        current_position=(200, 400),
        predicted_position=(500, 700),
        collision_zone=zone,
    )

    assert result is True


def test_center_path_misses_zone():

    zone = (
        350,
        550,
        650,
        1000,
    )

    result = check_path_collision(
        current_position=(100, 300),
        predicted_position=(200, 500),
        collision_zone=zone,
    )

    assert result is False


# ============================================================
# 3. ai2 bbox overlap 방식 테스트
# ============================================================

def test_bbox_overlap_ratio():

    zone = (
        350,
        550,
        650,
        1000,
    )

    bbox = (
        300,
        500,
        400,
        600,
    )

    ratio = (
        calculate_bbox_zone_overlap_ratio(
            bbox,
            zone,
        )
    )

    # 100x100 bbox 중
    # 50x50 영역이 zone과 겹침
    assert abs(
        ratio - 0.25
    ) < 1e-6


def test_future_bbox_translation():

    bbox = (
        100,
        100,
        200,
        200,
    )

    future_bbox = predict_future_bbox(
        current_bbox=bbox,
        current_position=(150, 150),
        predicted_position=(250, 200),
    )

    assert future_bbox == (
        200,
        150,
        300,
        250,
    )


def test_bbox_detects_collision_center_misses():

    zone = (
        400,
        500,
        600,
        800,
    )

    # 중심점은 zone 왼쪽을 지나지만,
    # bbox 오른쪽 부분은 zone과 겹치는 상황
    current_bbox = (
        250,
        450,
        390,
        590,
    )

    current_position = (
        320,
        520,
    )

    predicted_position = (
        350,
        650,
    )

    center_result = (
        check_path_collision(
            current_position,
            predicted_position,
            zone,
        )
    )

    bbox_result = (
        check_bbox_path_collision(
            current_bbox,
            current_position,
            predicted_position,
            zone,
        )
    )

    assert center_result is False
    assert bbox_result is True


# ============================================================
# 4. main bbox overlap 계산 테스트
# ============================================================

def test_partial_bbox_overlap():

    bbox = (
        25,
        42,
        45,
        58,
    )

    ratio = calculate_bbox_zone_overlap(
        bbox,
        ZONE,
    )

    # 교집합 = 5 * 16
    # bbox 면적 = 20 * 16
    # 겹침 비율 = 0.25
    assert ratio == pytest.approx(
        0.25
    )


def test_no_bbox_overlap():

    bbox = (
        0,
        0,
        10,
        10,
    )

    ratio = calculate_bbox_zone_overlap(
        bbox,
        ZONE,
    )

    assert ratio == 0.0


# ============================================================
# 5. main exact bbox collision 테스트
# ============================================================

def test_exact_bbox_overlaps_even_if_center_outside():

    bbox = (
        25,
        42,
        45,
        58,
    )

    center = (
        35,
        50,
    )

    # 기존 중심점 방식
    old_result = check_path_collision(
        center,
        center,
        ZONE,
    )

    # main exact bbox 방식
    exact_result = (
        check_bbox_path_collision_exact(
            bbox,
            center,
            center,
            ZONE,
        )
    )

    assert old_result is False
    assert exact_result is True


def test_exact_bbox_path_crosses_zone():

    bbox = (
        0,
        42,
        10,
        52,
    )

    current = (
        5,
        47,
    )

    predicted = (
        95,
        47,
    )

    result = (
        check_bbox_path_collision_exact(
            bbox,
            current,
            predicted,
            ZONE,
        )
    )

    assert result is True


def test_exact_bbox_path_misses_zone():

    bbox = (
        0,
        0,
        10,
        10,
    )

    current = (
        5,
        5,
    )

    predicted = (
        95,
        5,
    )

    result = (
        check_bbox_path_collision_exact(
            bbox,
            current,
            predicted,
            ZONE,
        )
    )

    assert result is False


def test_exact_fast_bbox_path_crosses_zone():

    bbox = (
        0,
        42,
        10,
        52,
    )

    current = (
        5,
        47,
    )

    predicted = (
        1000,
        47,
    )

    result = (
        check_bbox_path_collision_exact(
            bbox,
            current,
            predicted,
            ZONE,
        )
    )

    assert result is True


def test_exact_invalid_bbox():

    result = (
        check_bbox_path_collision_exact(
            bbox=(10, 10, 10, 20),
            current_position=(10, 15),
            predicted_position=(50, 15),
            collision_zone=ZONE,
        )
    )

    assert result is False