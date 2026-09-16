# ai2/approach.py


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


def calculate_approach_rate(history):
    """
    최근 두 프레임의 bbox 면적 변화율을 이용하여
    객체의 상대적인 접근 정도를 계산한다.

    반환값:
    초당 bbox 면적 변화 비율
    """

    if len(history) < 2:
        return 0.0

    previous = history[-2]
    current = history[-1]

    previous_area = calculate_bbox_area(previous["bbox"])
    current_area = calculate_bbox_area(current["bbox"])

    delta_time = current["timestamp"] - previous["timestamp"]

    if delta_time <= 0 or previous_area <= 0:
        return 0.0

    area_change_ratio = (
        current_area - previous_area
    ) / previous_area

    approach_rate = area_change_ratio / delta_time

    return approach_rate


def is_approaching(history, threshold=0.1):
    """
    ApproachRate가 threshold보다 크면
    접근 중인 객체로 판단한다.

    threshold는 임시값이며
    실제 촬영 데이터를 통해 추후 조정한다.
    """

    approach_rate = calculate_approach_rate(history)

    return approach_rate > threshold