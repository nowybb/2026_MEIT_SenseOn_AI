# ai2/approach.py

import math
from statistics import median


# ============================================================
# 1. Bounding Box 면적 계산
# ============================================================

def calculate_bbox_area(bbox):
    """
    Bounding Box의 면적을 계산한다.

    bbox 형식:
    [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = bbox

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


# ============================================================
# 2. ApproachRate 계산
# ============================================================

def calculate_approach_rate(
    history,
    window_size=10,
    min_samples=3,
    min_duration=0.1
):
    """
    최근 여러 기록의 bbox 면적 변화 추세를 이용하여
    객체의 상대적인 접근 정도를 계산한다.

    기존 방식:
        직전 두 기록의 면적 변화율

    개선 방식:
        1. 최근 N개 기록 추출
        2. bbox 면적 계산
        3. 로그 면적으로 변환
        4. 모든 유효한 시점 쌍의 기울기 계산
        5. 기울기 중앙값 사용

    반환값:
        양수: bbox 증가 추세
        음수: bbox 감소 추세
        0.0: 변화 없음 또는 데이터 부족

    단위:
        초당 로그 면적 변화율
    """

    # --------------------------------------------------------
    # 기록이 부족하면 계산하지 않음
    # --------------------------------------------------------

    if len(history) < min_samples:
        return 0.0

    # 최근 N개 기록
    recent_history = history[-window_size:]

    samples = []

    # --------------------------------------------------------
    # 유효한 timestamp와 bbox 면적 추출
    # --------------------------------------------------------

    for item in recent_history:

        timestamp = float(item["timestamp"])

        area = calculate_bbox_area(
            item["bbox"]
        )

        if not math.isfinite(timestamp):
            continue

        if not math.isfinite(area) or area <= 0:
            continue

        # 면적을 로그 스케일로 변환
        log_area = math.log(area)

        samples.append(
            (timestamp, log_area)
        )

    if len(samples) < min_samples:
        return 0.0

    # --------------------------------------------------------
    # 최소 관측 시간 확인
    # --------------------------------------------------------

    duration = (
        samples[-1][0] - samples[0][0]
    )

    if duration < min_duration:
        return 0.0

    # --------------------------------------------------------
    # 모든 시점 쌍의 기울기 계산
    # --------------------------------------------------------

    slopes = []

    for i in range(len(samples) - 1):

        time_i, area_i = samples[i]

        for j in range(i + 1, len(samples)):

            time_j, area_j = samples[j]

            delta_time = time_j - time_i

            if delta_time <= 0:
                continue

            slope = (
                area_j - area_i
            ) / delta_time

            slopes.append(slope)

    if not slopes:
        return 0.0

    # --------------------------------------------------------
    # 기울기 중앙값 사용
    # --------------------------------------------------------

    approach_rate = median(slopes)

    return approach_rate


# ============================================================
# 3. 접근 여부 판단
# ============================================================

def is_approaching(
    history,
    threshold=0.1,
    window_size=10
):
    """
    여러 프레임의 bbox 증가 추세를 기반으로
    접근 여부를 판단한다.

    threshold:
        초기 개발용 임계값

    실제 후방 영상과 Ground Truth를 이용하여
    추후 조정해야 한다.
    """

    approach_rate = calculate_approach_rate(
        history,
        window_size=window_size
    )

    return approach_rate > threshold