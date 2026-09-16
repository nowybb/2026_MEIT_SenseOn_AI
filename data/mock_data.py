# data/mock_data.py


def make_detection(
    track_id,
    class_name,
    timestamp,
    bbox,
    confidence=0.95,
):
    """
    AI1 출력 형식과 동일한 Mock Detection을 생성한다.
    """

    x1, y1, x2, y2 = bbox

    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": confidence,
        "timestamp": timestamp,
        "bbox": bbox,
        "center_x": (x1 + x2) / 2,
        "center_y": (y1 + y2) / 2,
    }


# =========================================================
# Scenario 1
# Collision Zone 방향으로 빠르게 접근하는 자동차
# 기대 결과: DANGER
# =========================================================

DANGER_APPROACH = [
    [
        make_detection(
            1, "car", 0.0,
            [300, 200, 380, 320]
        )
    ],
    [
        make_detection(
            1, "car", 0.1,
            [292, 192, 390, 332]
        )
    ],
    [
        make_detection(
            1, "car", 0.2,
            [280, 180, 405, 350]
        )
    ],
]


# =========================================================
# Scenario 2
# 크기 변화 없이 정지해 있는 자동차
# 기대 결과: SAFE
# =========================================================

STATIONARY_OBJECT = [
    [
        make_detection(
            2, "car", 0.0,
            [300, 200, 380, 320]
        )
    ],
    [
        make_detection(
            2, "car", 0.1,
            [300, 200, 380, 320]
        )
    ],
    [
        make_detection(
            2, "car", 0.2,
            [300, 200, 380, 320]
        )
    ],
]


# =========================================================
# Scenario 3
# 화면에서 점점 작아지는 자동차
# 기대 결과: SAFE
# =========================================================

MOVING_AWAY = [
    [
        make_detection(
            3, "car", 0.0,
            [270, 170, 410, 350]
        )
    ],
    [
        make_detection(
            3, "car", 0.1,
            [285, 185, 395, 335]
        )
    ],
    [
        make_detection(
            3, "car", 0.2,
            [300, 200, 380, 320]
        )
    ],
]


# =========================================================
# Scenario 4
# 왼쪽에서 오른쪽으로 이동하지만 bbox 크기는 동일
# Collision Zone으로 향하지 않음
# 기대 결과: SAFE
# =========================================================

SIDE_PASS = [
    [
        make_detection(
            4, "bicycle", 0.0,
            [80, 220, 130, 300]
        )
    ],
    [
        make_detection(
            4, "bicycle", 0.1,
            [90, 220, 140, 300]
        )
    ],
    [
        make_detection(
            4, "bicycle", 0.2,
            [100, 220, 150, 300]
        )
    ],
]


# =========================================================
# Scenario 5
# 실제로는 정지 상태지만 bbox가 조금씩 흔들림
# YOLO bbox jitter를 단순하게 재현
#
# 기대 결과: 최종적으로 SAFE
# =========================================================

BBOX_JITTER = [
    [
        make_detection(
            5, "car", 0.0,
            [300, 200, 380, 320]
        )
    ],
    [
        make_detection(
            5, "car", 0.1,
            [299, 199, 381, 321]
        )
    ],
    [
        make_detection(
            5, "car", 0.2,
            [301, 201, 379, 319]
        )
    ],
    [
        make_detection(
            5, "car", 0.3,
            [298, 198, 382, 322]
        )
    ],
    [
        make_detection(
            5, "car", 0.4,
            [300, 200, 380, 320]
        )
    ],
]


# =========================================================
# Scenario 6
# 위험 자동차 + 안전 자전거가 동시에 존재
#
# 기대 결과:
# car       -> DANGER
# bicycle   -> SAFE
# Primary   -> car
# =========================================================

MULTI_OBJECT = [
    [
        make_detection(
            10, "car", 0.0,
            [300, 200, 380, 320]
        ),
        make_detection(
            20, "bicycle", 0.0,
            [100, 220, 150, 300]
        ),
    ],
    [
        make_detection(
            10, "car", 0.1,
            [292, 192, 390, 332]
        ),
        make_detection(
            20, "bicycle", 0.1,
            [105, 220, 155, 300]
        ),
    ],
    [
        make_detection(
            10, "car", 0.2,
            [280, 180, 405, 350]
        ),
        make_detection(
            20, "bicycle", 0.2,
            [110, 220, 160, 300]
        ),
    ],
]


# 모든 테스트 시나리오
SCENARIOS = {
    "danger_approach": DANGER_APPROACH,
    "stationary_object": STATIONARY_OBJECT,
    "moving_away": MOVING_AWAY,
    "side_pass": SIDE_PASS,
    "bbox_jitter": BBOX_JITTER,
    "multi_object": MULTI_OBJECT,
}


# 기존 main.py와 호환하기 위해 유지
MOCK_FRAMES = MULTI_OBJECT