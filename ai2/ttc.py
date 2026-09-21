# ai2/ttc.py

import math

from ai2.approach import calculate_bbox_area


def calculate_visual_ttc(history, window_size=5):
    """
    Bounding Box의 영상상 크기 변화를 이용해
    Visual TTC(Time To Collision)를 추정한다.

    반환값:
    - 접근 중인 경우: TTC (초)
    - 접근하지 않거나 계산할 수 없는 경우: None
    """

    if len(history) < 2:
        return None

    recent_history = history[-window_size:]

    first = recent_history[0]
    last = recent_history[-1]

    delta_time = last["timestamp"] - first["timestamp"]

    if delta_time <= 0:
        return None

    first_area = calculate_bbox_area(first["bbox"])
    last_area = calculate_bbox_area(last["bbox"])

    if first_area <= 0 or last_area <= 0:
        return None

    # bbox 면적을 영상상 선형 크기로 변환
    first_scale = math.sqrt(first_area)
    last_scale = math.sqrt(last_area)

    scale_velocity = (
        last_scale - first_scale
    ) / delta_time

    # 영상상 크기가 증가하지 않으면 접근 중이 아니라고 판단
    if scale_velocity <= 0:
        return None

    # Visual TTC ≈ 현재 영상상 크기 / 크기 증가 속도
    visual_ttc = last_scale / scale_velocity

    return visual_ttc

def calculate_visual_ttc_robust(
    history,
    window_size=10,
    min_samples=3,
    min_duration=0.1,
):
    """
    최근 여러 프레임의 bbox 크기 변화를 이용해
    Visual TTC를 보다 안정적으로 추정한다.

    sqrt(bbox area)를 영상상 선형 크기로 사용하고,
    모든 유효한 시점 쌍의 scale 변화율을 계산한 뒤
    중앙값을 사용한다.

    반환값:
    - 접근 중인 경우: TTC (초)
    - 접근하지 않거나 계산할 수 없는 경우: None
    """

    if len(history) < min_samples:
        return None

    recent_history = history[-window_size:]

    samples = []

    for item in recent_history:

        timestamp = float(
            item["timestamp"]
        )

        area = calculate_bbox_area(
            item["bbox"]
        )

        if not math.isfinite(timestamp):
            continue

        if (
            not math.isfinite(area)
            or area <= 0
        ):
            continue

        scale = math.sqrt(area)

        samples.append(
            (timestamp, scale)
        )

    if len(samples) < min_samples:
        return None

    duration = (
        samples[-1][0]
        - samples[0][0]
    )

    if duration < min_duration:
        return None

    scale_velocities = []

    for i in range(
        len(samples) - 1
    ):

        time_i, scale_i = (
            samples[i]
        )

        for j in range(
            i + 1,
            len(samples)
        ):

            time_j, scale_j = (
                samples[j]
            )

            delta_time = (
                time_j - time_i
            )

            if delta_time <= 0:
                continue

            scale_velocity = (
                scale_j - scale_i
            ) / delta_time

            if math.isfinite(
                scale_velocity
            ):
                scale_velocities.append(
                    scale_velocity
                )

    if not scale_velocities:
        return None

    scale_velocity = sorted(
        scale_velocities
    )[len(scale_velocities) // 2]

    # 영상상 크기가 증가하지 않으면 접근하지 않는 것으로 판단
    if scale_velocity <= 0:
        return None

    current_scale = (
        samples[-1][1]
    )

    visual_ttc = (
        current_scale
        / scale_velocity
    )

    if (
        not math.isfinite(visual_ttc)
        or visual_ttc <= 0
    ):
        return None

    return visual_ttc