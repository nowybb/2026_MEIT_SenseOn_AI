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

def is_point_in_zone(x, y, collision_zone):

    x_min, y_min, x_max, y_max = collision_zone

    return (
        x_min <= x <= x_max
        and y_min <= y <= y_max
    )


# ============================================================
# 3. 중심점 기반 경로 교차 검사
# 기존 함수 유지
# ============================================================

def check_path_collision(
    current_position,
    predicted_position,
    collision_zone,
    num_samples=20,
):
    """
    현재 중심점부터 예측 중심점까지의 경로가
    Collision Zone과 교차하는지 확인
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


# ============================================================
# 4. bbox와 Collision Zone의 겹침 비율
# 신규 함수
# ============================================================

def calculate_bbox_zone_overlap(
    bbox,
    collision_zone
):
    """
    bbox 중 Collision Zone과 겹치는 면적 비율 계산

    overlap_ratio = 교집합 면적 / bbox 면적

    반환 범위:
        0.0 ~ 1.0

    주의:
        공간적 겹침 비율이며 실제 충돌 확률이 아님
    """

    x1, y1, x2, y2 = bbox

    zx1, zy1, zx2, zy2 = collision_zone

    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)

    bbox_area = bbox_width * bbox_height

    # 유효하지 않은 bbox
    if bbox_area <= 0:
        return 0.0

    # 두 사각형의 교집합
    inter_x1 = max(x1, zx1)
    inter_y1 = max(y1, zy1)

    inter_x2 = min(x2, zx2)
    inter_y2 = min(y2, zy2)

    inter_width = max(
        0.0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1
    )

    intersection_area = (
        inter_width * inter_height
    )

    overlap_ratio = (
        intersection_area / bbox_area
    )

    return overlap_ratio


# ============================================================
# 5. 선분과 사각형의 교차 검사
# 신규 내부 함수
# ============================================================

def _segment_intersects_rectangle(
    start,
    end,
    rectangle
):
    """
    시작점~끝점 선분이 사각형과 교차하는지 확인

    선분의 매개변수 범위 0~1을 이용한
    정확한 축 정렬 사각형 교차 검사
    """

    x1, y1 = start
    x2, y2 = end

    rx1, ry1, rx2, ry2 = rectangle

    dx = x2 - x1
    dy = y2 - y1

    t_min = 0.0
    t_max = 1.0

    # x축, y축을 각각 검사
    for position, velocity, lower, upper in (
        (x1, dx, rx1, rx2),
        (y1, dy, ry1, ry2),
    ):

        # 해당 축에서 움직이지 않는 경우
        if velocity == 0:

            if position < lower or position > upper:
                return False

            continue

        t1 = (lower - position) / velocity
        t2 = (upper - position) / velocity

        enter = min(t1, t2)
        exit = max(t1, t2)

        t_min = max(t_min, enter)
        t_max = min(t_max, exit)

        if t_min > t_max:
            return False

    return True


# ============================================================
# 6. bbox 크기를 반영한 예측 경로 검사
# 신규 함수
# ============================================================

def check_bbox_path_collision(
    bbox,
    current_position,
    predicted_position,
    collision_zone
):
    """
    bbox가 현재 위치에서 예측 위치까지
    평행 이동한다고 가정했을 때
    Collision Zone과 교차하는지 확인

    중심점만 검사하는 기존 방식과 달리
    bbox의 가로/세로 크기를 반영

    주의:
        - bbox 크기는 예측 구간에서 일정하다고 가정
        - 실제 충돌 여부를 확정하는 함수가 아님
        - 원근법 기반 근접도는 별도 계산 필요
    """

    if current_position is None or predicted_position is None:
        return False

    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    if bbox_width <= 0 or bbox_height <= 0:
        return False

    current_x, current_y = current_position

    zx1, zy1, zx2, zy2 = collision_zone

    # 현재 중심점을 기준으로 bbox의 각 방향 크기
    left_extent = current_x - x1
    right_extent = x2 - current_x

    top_extent = current_y - y1
    bottom_extent = y2 - current_y

    # bbox 크기만큼 Collision Zone을 확장
    #
    # bbox와 Zone의 교차 문제를
    # 중심점과 확장된 Zone의 교차 문제로 변환
    expanded_zone = (
        zx1 - right_extent,
        zy1 - bottom_extent,
        zx2 + left_extent,
        zy2 + top_extent
    )

    # 현재~예측 중심점의 경로 검사
    return _segment_intersects_rectangle(
        current_position,
        predicted_position,
        expanded_zone
    )