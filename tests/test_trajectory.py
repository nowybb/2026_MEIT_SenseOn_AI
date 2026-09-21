from ai2.trajectory import (
    calculate_velocity,
    calculate_velocity_robust,
    predict_future_position,
    predict_future_position_robust,
)


def make_detection(timestamp, x, y):
    return {
        "timestamp": timestamp,
        "center_x": x,
        "center_y": y,
    }


# ============================================================
# 1. 일정한 직선 이동
# ============================================================

def test_constant_linear_motion():

    history = [
        make_detection(0.0, 100, 100),
        make_detection(0.1, 105, 102),
        make_detection(0.2, 110, 104),
        make_detection(0.3, 115, 106),
        make_detection(0.4, 120, 108),
    ]

    vx_old, vy_old = calculate_velocity(history)
    vx_new, vy_new = calculate_velocity_robust(history)

    assert abs(vx_old - 50.0) < 1e-6
    assert abs(vy_old - 20.0) < 1e-6

    assert abs(vx_new - 50.0) < 1e-6
    assert abs(vy_new - 20.0) < 1e-6


# ============================================================
# 2. 정지 객체
# ============================================================

def test_stationary_object():

    history = [
        make_detection(i * 0.1, 320, 240)
        for i in range(6)
    ]

    vx, vy = calculate_velocity_robust(history)

    assert abs(vx) < 1e-9
    assert abs(vy) < 1e-9


# ============================================================
# 3. 정상 이동 중 한 프레임 위치가 튀는 경우
# ============================================================

def test_position_outlier():

    history = [
        make_detection(0.0, 100, 100),
        make_detection(0.1, 105, 100),
        make_detection(0.2, 110, 100),

        # Detection 중심점이 순간적으로 튄 상황
        make_detection(0.3, 150, 100),

        make_detection(0.4, 120, 100),
        make_detection(0.5, 125, 100),
        make_detection(0.6, 130, 100),
    ]

    vx, vy = calculate_velocity_robust(history)

    # 실제 이동 추세는 초당 약 50 pixel
    assert abs(vx - 50.0) < 1e-6
    assert abs(vy) < 1e-9


# ============================================================
# 4. 0.3초 후 위치 예측
# ============================================================

def test_robust_future_prediction():

    history = [
        make_detection(0.0, 100, 100),
        make_detection(0.1, 105, 100),
        make_detection(0.2, 110, 100),
        make_detection(0.3, 115, 100),
        make_detection(0.4, 120, 100),
    ]

    predicted = predict_future_position_robust(
        history,
        future_time=0.3
    )

    predicted_x, predicted_y = predicted

    # vx = 50 pixel/s
    # 0.3초 후 = 15 pixel 이동
    assert abs(predicted_x - 135.0) < 1e-6
    assert abs(predicted_y - 100.0) < 1e-6