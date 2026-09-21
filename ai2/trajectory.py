# ai2/trajectory.py


def calculate_velocity(history, window_size=5):
    """
    최근 N개 기록의 중심점 변화를 이용하여
    객체의 영상 내 이동 속도를 계산한다.

    반환값:
    vx, vy (pixel / second)
    """

    if len(history) < 2:
        return 0.0, 0.0

    recent_history = history[-window_size:]

    first = recent_history[0]
    last = recent_history[-1]

    delta_time = last["timestamp"] - first["timestamp"]

    if delta_time <= 0:
        return 0.0, 0.0

    vx = (
        last["center_x"] - first["center_x"]
    ) / delta_time

    vy = (
        last["center_y"] - first["center_y"]
    ) / delta_time

    return vx, vy


def predict_future_position(history, future_time=1.0, window_size=5):
    """
    현재 이동 속도가 유지된다고 가정하고
    future_time초 뒤의 중심 위치를 예측한다.

    반환값:
    predicted_x, predicted_y
    """

    if not history:
        return None

    current = history[-1]

    vx, vy = calculate_velocity(
        history,
        window_size=window_size
    )

    predicted_x = current["center_x"] + vx * future_time
    predicted_y = current["center_y"] + vy * future_time

    return predicted_x, predicted_y

from statistics import median


def calculate_velocity_robust(history, window_size=10):
    """
    최근 N개 기록에서 연속된 시점별 속도를 계산하고,
    vx와 vy의 중앙값을 사용하여
    순간적인 중심점 흔들림의 영향을 줄인다.

    반환값:
        vx, vy (pixel / second)
    """

    if len(history) < 2:
        return 0.0, 0.0

    recent_history = history[-window_size:]

    vx_samples = []
    vy_samples = []

    for i in range(1, len(recent_history)):

        previous = recent_history[i - 1]
        current = recent_history[i]

        delta_time = (
            current["timestamp"]
            - previous["timestamp"]
        )

        if delta_time <= 0:
            continue

        vx = (
            current["center_x"]
            - previous["center_x"]
        ) / delta_time

        vy = (
            current["center_y"]
            - previous["center_y"]
        ) / delta_time

        vx_samples.append(vx)
        vy_samples.append(vy)

    if not vx_samples:
        return 0.0, 0.0

    return (
        median(vx_samples),
        median(vy_samples),
    )


def predict_future_position_robust(
    history,
    future_time=0.3,
    window_size=10
):
    """
    중앙값 기반 속도를 사용하여
    짧은 시간 뒤의 위치를 예측한다.

    기본 예측 시간:
        0.3초
    """

    if not history:
        return None

    current = history[-1]

    vx, vy = calculate_velocity_robust(
        history,
        window_size=window_size
    )

    predicted_x = (
        current["center_x"]
        + vx * future_time
    )

    predicted_y = (
        current["center_y"]
        + vy * future_time
    )

    return predicted_x, predicted_y