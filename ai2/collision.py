# ai2/collision.py


# ============================================================
# 1. Collision Zone 생성
# 기존 함수 유지
# ============================================================

def create_collision_zone(
    frame_width,
    frame_height,
    x_min_ratio=0.35,
    x_max_ratio=0.65,
    y_min_ratio=0.55,
    y_max_ratio=1.0,
):
    """
    화면 크기를 기준으로 Collision Zone 생성

    반환값:
    (x_min, y_min, x_max, y_max)
    """

    x_min = frame_width * x_min_ratio
    x_max = frame_width * x_max_ratio

    y_min = frame_height * y_min_ratio
    y_max = frame_height * y_max_ratio

    return x_min, y_min, x_max, y_max


# ============================================================
# 2. 중심점의 Collision Zone 포함 여부
# 기존 함수 유지
# ============================================================

def is_point_in_zone(
    x,
    y,
    collision_zone,
):
    """
    하나의 점이 Collision Zone 내부에 있는지 확인한다.
    """

    x_min, y_min, x_max, y_max = collision_zone

    return (
        x_min <= x <= x_max
        and y_min <= y <= y_max
    )


# ============================================================
# 3. 중심점 기반 경로 교차 검사
# 기존 baseline
# ============================================================

def check_path_collision(
    current_position,
    predicted_position,
    collision_zone,
    num_samples=20,
):
    """
    현재 중심점부터 예측 중심점까지의 경로가
    Collision Zone과 교차하는지 확인한다.

    기존 center 기반 baseline 방식.
    """

    if (
        current_position is None
        or predicted_position is None
    ):
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
            collision_zone,
        ):
            return True

    return False


# ============================================================
# 4. bbox와 Collision Zone의 겹침 비율
# ai2 후보 방식
# ============================================================

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
        x2 - x1,
    )

    bbox_height = max(
        0.0,
        y2 - y1,
    )

    bbox_area = (
        bbox_width
        * bbox_height
    )

    if bbox_area <= 0:
        return 0.0

    overlap_x1 = max(
        x1,
        zone_x1,
    )

    overlap_y1 = max(
        y1,
        zone_y1,
    )

    overlap_x2 = min(
        x2,
        zone_x2,
    )

    overlap_y2 = min(
        y2,
        zone_y2,
    )

    overlap_width = max(
        0.0,
        overlap_x2 - overlap_x1,
    )

    overlap_height = max(
        0.0,
        overlap_y2 - overlap_y1,
    )

    overlap_area = (
        overlap_width
        * overlap_height
    )

    return (
        overlap_area
        / bbox_area
    )


# ============================================================
# 5. 미래 bbox 위치 계산
# ai2 후보 방식
# ============================================================

def predict_future_bbox(
    current_bbox,
    current_position,
    predicted_position,
):
    """
    현재 bbox의 크기는 유지한 채,
    중심점 이동량만큼 bbox 전체를 이동시킨다.

    주의:
        bbox 크기 변화는 고려하지 않는다.
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


# ============================================================
# 6. bbox overlap 기반 경로 검사
# ai2에서 실제 영상 비교에 사용한 후보
# ============================================================

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

    주의:
        overlap_threshold는 개발용 초기값이며
        Ground Truth 평가 후 조정할 수 있다.
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


# ============================================================
# 7. bbox와 Collision Zone의 겹침 비율
# main에서 추가된 방식
# ============================================================

def calculate_bbox_zone_overlap(
    bbox,
    collision_zone,
):
    """
    bbox 중 Collision Zone과 겹치는 면적 비율을 계산한다.

    overlap_ratio = 교집합 면적 / bbox 면적

    반환 범위:
        0.0 ~ 1.0

    주의:
        공간적 겹침 비율이며 실제 충돌 확률이 아니다.
    """

    if bbox is None:
        return 0.0

    x1, y1, x2, y2 = bbox

    zx1, zy1, zx2, zy2 = collision_zone

    bbox_width = max(
        0.0,
        x2 - x1,
    )

    bbox_height = max(
        0.0,
        y2 - y1,
    )

    bbox_area = (
        bbox_width
        * bbox_height
    )

    if bbox_area <= 0:
        return 0.0

    inter_x1 = max(
        x1,
        zx1,
    )

    inter_y1 = max(
        y1,
        zy1,
    )

    inter_x2 = min(
        x2,
        zx2,
    )

    inter_y2 = min(
        y2,
        zy2,
    )

    inter_width = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection_area = (
        inter_width
        * inter_height
    )

    overlap_ratio = (
        intersection_area
        / bbox_area
    )

    return overlap_ratio


# ============================================================
# 8. 선분과 사각형의 정확한 교차 검사
# main에서 추가된 내부 함수
# ============================================================

def _segment_intersects_rectangle(
    start,
    end,
    rectangle,
):
    """
    시작점~끝점 선분이 사각형과 교차하는지 확인한다.

    선분의 매개변수 범위 0~1을 이용한
    축 정렬 사각형 교차 검사.
    """

    x1, y1 = start
    x2, y2 = end

    rx1, ry1, rx2, ry2 = rectangle

    dx = x2 - x1
    dy = y2 - y1

    t_min = 0.0
    t_max = 1.0

    for position, velocity, lower, upper in (
        (x1, dx, rx1, rx2),
        (y1, dy, ry1, ry2),
    ):

        # 해당 축에서 움직이지 않는 경우
        if velocity == 0:

            if (
                position < lower
                or position > upper
            ):
                return False

            continue

        t1 = (
            lower - position
        ) / velocity

        t2 = (
            upper - position
        ) / velocity

        enter = min(
            t1,
            t2,
        )

        exit = max(
            t1,
            t2,
        )

        t_min = max(
            t_min,
            enter,
        )

        t_max = min(
            t_max,
            exit,
        )

        if t_min > t_max:
            return False

    return True


# ============================================================
# 9. bbox 크기를 반영한 정확한 경로 교차 검사
# main에서 추가된 후보
# ============================================================

def check_bbox_path_collision_exact(
    bbox,
    current_position,
    predicted_position,
    collision_zone,
):
    """
    bbox가 현재 위치에서 예측 위치까지
    평행 이동한다고 가정했을 때
    Collision Zone과 교차하는지 확인한다.

    Collision Zone을 bbox 크기만큼 확장한 뒤,
    현재 중심점과 예측 중심점을 연결한 선분이
    확장된 Zone과 교차하는지 검사한다.

    중심점만 검사하는 기존 방식과 달리
    bbox의 가로/세로 크기를 반영한다.

    주의:
        - bbox 크기는 예측 구간에서 일정하다고 가정
        - 실제 충돌 여부를 확정하는 함수가 아님
        - overlap threshold는 사용하지 않음
        - 원근법 기반 근접도는 별도 계산 필요
    """

    if (
        bbox is None
        or current_position is None
        or predicted_position is None
    ):
        return False

    x1, y1, x2, y2 = bbox

    bbox_width = (
        x2 - x1
    )

    bbox_height = (
        y2 - y1
    )

    if (
        bbox_width <= 0
        or bbox_height <= 0
    ):
        return False

    current_x, current_y = (
        current_position
    )

    zx1, zy1, zx2, zy2 = (
        collision_zone
    )

    # 현재 중심점을 기준으로 bbox의 각 방향 크기
    left_extent = (
        current_x - x1
    )

    right_extent = (
        x2 - current_x
    )

    top_extent = (
        current_y - y1
    )

    bottom_extent = (
        y2 - current_y
    )

    # bbox 크기만큼 Collision Zone을 확장
    #
    # bbox와 Zone의 교차 문제를
    # 중심점과 확장된 Zone의 교차 문제로 변환
    expanded_zone = (
        zx1 - right_extent,
        zy1 - bottom_extent,
        zx2 + left_extent,
        zy2 + top_extent,
    )

    return _segment_intersects_rectangle(
        current_position,
        predicted_position,
        expanded_zone,
    )