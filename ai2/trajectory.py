# ai2/trajectory.py

import math
from statistics import median


# ============================================================
# 1. 기존 속도 계산
# ============================================================

def calculate_velocity(history, window_size=5):
    """
    최근 N개 기록의 처음과 마지막 중심점을 이용해
    영상 내 이동 속도를 계산한다.

    반환값:
        vx, vy (pixel / second)

    주의:
        실제 차량의 주행 속도나 방향이 아니라
        영상 좌표계에서의 중심점 변화량이다.
    """

    if len(history) < 2:
        return 0.0, 0.0

    recent_history = history[-window_size:]

    first = recent_history[0]
    last = recent_history[-1]

    delta_time = (
        last["timestamp"] - first["timestamp"]
    )

    if delta_time <= 0:
        return 0.0, 0.0

    vx = (
        last["center_x"] - first["center_x"]
    ) / delta_time

    vy = (
        last["center_y"] - first["center_y"]
    ) / delta_time

    return vx, vy


# ============================================================
# 2. 기존 미래 위치 예측
# ============================================================

def predict_future_position(
    history,
    future_time=1.0,
    window_size=5
):
    """
    기존 방식:
        최근 5개 기록의 양 끝을 이용한 속도 추정
        → 1초 후 위치 예측

    현재 main_video.py에서 사용하는 함수다.
    비교를 위해 기존 동작을 유지한다.
    """

    if not history:
        return None

    current = history[-1]

    vx, vy = calculate_velocity(
        history,
        window_size=window_size
    )

    predicted_x = (
        current["center_x"] + vx * future_time
    )

    predicted_y = (
        current["center_y"] + vy * future_time
    )

    return predicted_x, predicted_y


# ============================================================
# 3. 개선 후보: 여러 시점 기반 속도 계산
# ============================================================

def calculate_stable_velocity(
    history,
    window_size=10,
    min_pair_duration=0.08
):
    """
    개선 후보:
        최근 최대 10개 기록에서 여러 시점 쌍의
        중심점 변화율을 계산한다.

        각 축의 변화율 중앙값을 대표 속도로 사용한다.

    반환값:
        (vx, vy): 추정 성공
        None: 기록이 부족하거나 계산 불가능

    주의:
        bbox 중심점의 일시적 흔들림을 완화하기 위한
        방법이며 실제 차량의 주행 방향을 보장하지 않는다.
    """

    recent_history = history[-window_size:]

    # 계산에 필요한 최소 기록 수
    if len(recent_history) < 3:
        return None

    total_duration = (
        recent_history[-1]["timestamp"]
        - recent_history[0]["timestamp"]
    )

    # 충분한 시간 간격이 확보되지 않은 경우
    if (
        not math.isfinite(total_duration)
        or total_duration < 0.1
    ):
        return None

    vx_values = []
    vy_values = []

    # 여러 시점 쌍의 속도 계산
    for i in range(len(recent_history)):

        for j in range(i + 1, len(recent_history)):

            first = recent_history[i]
            last = recent_history[j]

            delta_time = (
                last["timestamp"]
                - first["timestamp"]
            )

            # 너무 짧거나 잘못된 시간 간격 제외
            if (
                not math.isfinite(delta_time)
                or delta_time < min_pair_duration
            ):
                continue

            vx = (
                last["center_x"]
                - first["center_x"]
            ) / delta_time

            vy = (
                last["center_y"]
                - first["center_y"]
            ) / delta_time

            # 계산 결과가 유효한 경우만 사용
            if (
                math.isfinite(vx)
                and math.isfinite(vy)
            ):
                vx_values.append(vx)
                vy_values.append(vy)

    # 유효한 속도 계산값이 없는 경우
    if not vx_values:
        return None

    stable_vx = median(vx_values)
    stable_vy = median(vy_values)

    return stable_vx, stable_vy


# ============================================================
# 4. 개선 후보: 안정화된 미래 위치 예측
# ============================================================

def predict_stable_position(
    history,
    future_time=0.3,
    window_size=10
):
    """
    개선 후보:
        여러 시점에서 계산한 대표 속도를 이용해
        짧은 시간 뒤의 위치를 예측한다.

    기본 예측 시간:
        0.3초

    반환값:
        (predicted_x, predicted_y): 예측 성공
        None: 예측 불가능

    주의:
        기존 위험도 판단에 바로 연결하지 않고
        비교용으로만 사용한다.
    """

    if not history:
        return None

    velocity = calculate_stable_velocity(
        history,
        window_size=window_size
    )

    # 기록 부족 등으로 속도를 추정할 수 없는 경우
    if velocity is None:
        return None

    vx, vy = velocity

    current = history[-1]

    predicted_x = (
        current["center_x"] + vx * future_time
    )

    predicted_y = (
        current["center_y"] + vy * future_time
    )

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