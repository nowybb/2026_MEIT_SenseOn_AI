# ai2/risk.py


# 개발용 초기 임계값
# 실제 촬영 및 Ground Truth 평가 후 조정 예정
DANGER_TTC_THRESHOLD = 2.0
CAUTION_TTC_THRESHOLD = 4.0


RISK_PRIORITY = {
    "SAFE": 0,
    "CAUTION": 1,
    "DANGER": 2,
}


def determine_risk_level(
    approaching,
    path_collision,
    visual_ttc,
):
    """
    접근 여부, 예상 경로 충돌 여부, Visual TTC를 이용해
    위험 수준을 판단한다.

    반환값:
    SAFE / CAUTION / DANGER
    """

    # 접근하지 않는 객체
    if not approaching:
        return "SAFE"

    # 접근하고 있지만 예상 이동 경로가
    # Collision Zone과 겹치지 않는 경우
    if not path_collision:
        return "SAFE"

    # 접근 + 경로 충돌이지만
    # TTC를 계산할 수 없는 경우
    if visual_ttc is None:
        return "CAUTION"

    # TTC가 매우 짧은 경우
    if visual_ttc <= DANGER_TTC_THRESHOLD:
        return "DANGER"

    # TTC가 비교적 짧은 경우
    if visual_ttc <= CAUTION_TTC_THRESHOLD:
        return "CAUTION"

    return "SAFE"


def select_primary_hazard(hazards):
    """
    여러 객체 중 가장 우선적으로 경고해야 할 객체를 선정한다.

    우선순위:
    1. 위험 수준이 높은 객체
    2. 같은 위험 수준이면 Visual TTC가 짧은 객체

    hazards 예시:
    [
        {
            "track_id": 1,
            "risk_level": "DANGER",
            "visual_ttc": 1.5
        },
        ...
    ]
    """

    if not hazards:
        return None

    non_safe_hazards = [
        hazard
        for hazard in hazards
        if hazard["risk_level"] != "SAFE"
    ]

    if not non_safe_hazards:
        return None

    def sort_key(hazard):
        risk_priority = RISK_PRIORITY[
            hazard["risk_level"]
        ]

        visual_ttc = hazard["visual_ttc"]

        # TTC 계산 불가능 객체는
        # 같은 위험 단계에서 후순위로 처리
        if visual_ttc is None:
            visual_ttc = float("inf")

        return (
            -risk_priority,
            visual_ttc,
        )

    return sorted(
        non_safe_hazards,
        key=sort_key
    )[0]