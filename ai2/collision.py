# ai2/collision.py


def create_collision_zone(
    frame_width,
    frame_height,
    x_min_ratio=0.35,
    x_max_ratio=0.65,
    y_min_ratio=0.55,
    y_max_ratio=1.0,
):
    """
    화면 크기를 기준으로 Collision Zone을 생성한다.

    반환값:
    (x_min, y_min, x_max, y_max)
    """

    x_min = frame_width * x_min_ratio
    x_max = frame_width * x_max_ratio

    y_min = frame_height * y_min_ratio
    y_max = frame_height * y_max_ratio

    return x_min, y_min, x_max, y_max


def is_point_in_zone(x, y, collision_zone):
    """
    하나의 점이 Collision Zone 내부에 있는지 확인한다.
    """

    x_min, y_min, x_max, y_max = collision_zone

    return (
        x_min <= x <= x_max
        and y_min <= y <= y_max
    )


def check_path_collision(
    current_position,
    predicted_position,
    collision_zone,
    num_samples=20,
):
    """
    현재 위치부터 미래 예측 위치까지의 직선 경로를
    여러 점으로 나누어 Collision Zone 진입 여부를 확인한다.

    하나라도 Collision Zone 내부에 들어오면 True를 반환한다.
    """

    if current_position is None or predicted_position is None:
        return False

    current_x, current_y = current_position
    predicted_x, predicted_y = predicted_position

    for i in range(num_samples + 1):

        ratio = i / num_samples

        sample_x = (
            current_x
            + (predicted_x - current_x) * ratio
        )

        sample_y = (
            current_y
            + (predicted_y - current_y) * ratio
        )

        if is_point_in_zone(
            sample_x,
            sample_y,
            collision_zone
        ):
            return True

    return False

def calculate_bbox_zone_overlap_ratio(
    bbox,
    collision_zone,
):
    """
    Bounding Box와 Collision Zone의 겹치는 비율을 계산한다.

    기준:
        overlap area / bbox area

    반환값:
        0.0 ~ 1.0
    """

    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox

    zone_x1, zone_y1, zone_x2, zone_y2 = (
        collision_zone
    )

    bbox_width = max(
        0.0,
        x2 - x1
    )

    bbox_height = max(
        0.0,
        y2 - y1
    )

    bbox_area = (
        bbox_width
        * bbox_height
    )

    if bbox_area <= 0:
        return 0.0

    overlap_x1 = max(
        x1,
        zone_x1
    )

    overlap_y1 = max(
        y1,
        zone_y1
    )

    overlap_x2 = min(
        x2,
        zone_x2
    )

    overlap_y2 = min(
        y2,
        zone_y2
    )

    overlap_width = max(
        0.0,
        overlap_x2 - overlap_x1
    )

    overlap_height = max(
        0.0,
        overlap_y2 - overlap_y1
    )

    overlap_area = (
        overlap_width
        * overlap_height
    )

    return (
        overlap_area
        / bbox_area
    )


def predict_future_bbox(
    current_bbox,
    current_position,
    predicted_position,
):
    """
    현재 bbox의 크기는 유지한 채,
    중심점 이동량만큼 bbox 전체를 이동시킨다.
    """

    if (
        current_bbox is None
        or current_position is None
        or predicted_position is None
    ):
        return None

    x1, y1, x2, y2 = current_bbox

    current_x, current_y = (
        current_position
    )

    predicted_x, predicted_y = (
        predicted_position
    )

    dx = (
        predicted_x
        - current_x
    )

    dy = (
        predicted_y
        - current_y
    )

    return (
        x1 + dx,
        y1 + dy,
        x2 + dx,
        y2 + dy,
    )


def check_bbox_path_collision(
    current_bbox,
    current_position,
    predicted_position,
    collision_zone,
    num_samples=20,
    overlap_threshold=0.05,
):
    """
    현재 bbox부터 미래 예측 bbox까지 이동 경로를 샘플링하여
    Collision Zone과 bbox가 일정 비율 이상 겹치는지 확인한다.

    overlap_threshold=0.05:
        bbox 면적의 5% 이상이 Collision Zone과 겹치면
        충돌 경로 후보로 판단한다.
    """

    if (
        current_bbox is None
        or current_position is None
        or predicted_position is None
    ):
        return False

    future_bbox = predict_future_bbox(
        current_bbox,
        current_position,
        predicted_position,
    )

    if future_bbox is None:
        return False

    current_x1, current_y1, current_x2, current_y2 = (
        current_bbox
    )

    future_x1, future_y1, future_x2, future_y2 = (
        future_bbox
    )

    for i in range(num_samples + 1):

        ratio = (
            i / num_samples
        )

        sample_bbox = (

            current_x1
            + (
                future_x1
                - current_x1
            ) * ratio,

            current_y1
            + (
                future_y1
                - current_y1
            ) * ratio,

            current_x2
            + (
                future_x2
                - current_x2
            ) * ratio,

            current_y2
            + (
                future_y2
                - current_y2
            ) * ratio,
        )

        overlap_ratio = (
            calculate_bbox_zone_overlap_ratio(
                sample_bbox,
                collision_zone,
            )
        )

        if (
            overlap_ratio
            >= overlap_threshold
        ):
            return True

    return False