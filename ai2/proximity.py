def calculate_proximity_features(
    bbox,
    frame_width,
    frame_height
):
    """
    bbox와 영상 크기를 이용하여
    상대적 근접도 판단에 필요한 특징을 계산한다.

    반환값:
        height_ratio
        area_ratio
        bottom_y_ratio

    주의:
        실제 거리(m)를 계산하는 함수가 아님.
        단독으로 SAFE/DANGER를 결정하지 않음.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("영상 크기는 양수여야 합니다.")

    x1, y1, x2, y2 = bbox

    # 잘못된 bbox 형식 확인
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox 좌표 순서가 올바르지 않습니다.")

    # 영상 범위 안으로 좌표 제한
    x1 = max(0, min(x1, frame_width))
    x2 = max(0, min(x2, frame_width))

    y1 = max(0, min(y1, frame_height))
    y2 = max(0, min(y2, frame_height))

    # bbox 크기
    bbox_width = max(0, x2 - x1)
    bbox_height = max(0, y2 - y1)

    bbox_area = bbox_width * bbox_height
    frame_area = frame_width * frame_height

    # 1. 정규화된 bbox 높이
    height_ratio = bbox_height / frame_height

    # 2. 정규화된 bbox 면적
    area_ratio = bbox_area / frame_area

    # 3. 정규화된 bbox 하단 위치
    bottom_y_ratio = y2 / frame_height

    return {
        "height_ratio": height_ratio,
        "area_ratio": area_ratio,
        "bottom_y_ratio": bottom_y_ratio
    }