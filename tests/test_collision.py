from ai2.collision import (
    create_collision_zone,
    is_point_in_zone,
    check_path_collision,
    calculate_bbox_zone_overlap_ratio,
    predict_future_bbox,
    check_bbox_path_collision,
)


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