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