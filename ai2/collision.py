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