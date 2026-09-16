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