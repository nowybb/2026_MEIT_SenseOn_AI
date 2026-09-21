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
    
class RiskStabilizer:
    """
    프레임 간 Risk Level 변화를 안정화한다.

    원칙:
    - 위험 수준 상승은 즉시 반영
    - 위험 수준 하락은 일정 횟수 연속 확인 후 반영

    예:
        SAFE -> DANGER
            즉시 DANGER

        DANGER -> CAUTION
            CAUTION이 release_frames회 연속 발생해야 하락

        DANGER -> SAFE
            SAFE가 release_frames회 연속 발생해야 하락
    """

    def __init__(self, release_frames=3):

        if release_frames < 1:
            raise ValueError(
                "release_frames는 1 이상이어야 합니다."
            )

        self.release_frames = release_frames

        # track_id별 안정화 상태
        self.states = {}

    def update(
        self,
        track_id,
        raw_risk_level,
    ):
        """
        객체 하나의 raw risk를 받아
        안정화된 risk level을 반환한다.
        """

        if raw_risk_level not in RISK_PRIORITY:
            raise ValueError(
                f"알 수 없는 risk level: {raw_risk_level}"
            )

        # 처음 등장한 객체
        if track_id not in self.states:

            self.states[track_id] = {
                "stable_level": raw_risk_level,
                "pending_level": None,
                "pending_count": 0,
            }

            return raw_risk_level

        state = self.states[
            track_id
        ]

        stable_level = state[
            "stable_level"
        ]

        stable_priority = (
            RISK_PRIORITY[
                stable_level
            ]
        )

        raw_priority = (
            RISK_PRIORITY[
                raw_risk_level
            ]
        )

        # ---------------------------------------------
        # 동일한 위험 수준
        # ---------------------------------------------

        if raw_risk_level == stable_level:

            state[
                "pending_level"
            ] = None

            state[
                "pending_count"
            ] = 0

            return stable_level

        # ---------------------------------------------
        # 위험 상승
        # 즉시 반영
        # ---------------------------------------------

        if raw_priority > stable_priority:

            state[
                "stable_level"
            ] = raw_risk_level

            state[
                "pending_level"
            ] = None

            state[
                "pending_count"
            ] = 0

            return raw_risk_level

        # ---------------------------------------------
        # 위험 하락
        # 일정 프레임 연속 확인
        # ---------------------------------------------

        if (
            state["pending_level"]
            == raw_risk_level
        ):

            state[
                "pending_count"
            ] += 1

        else:

            state[
                "pending_level"
            ] = raw_risk_level

            state[
                "pending_count"
            ] = 1

        if (
            state["pending_count"]
            >= self.release_frames
        ):

            state[
                "stable_level"
            ] = raw_risk_level

            state[
                "pending_level"
            ] = None

            state[
                "pending_count"
            ] = 0

        return state[
            "stable_level"
        ]

    def remove_track(
        self,
        track_id,
    ):
        """
        더 이상 사용하지 않는 track 상태를 제거한다.
        """

        self.states.pop(
            track_id,
            None,
        )

    def reset(self):
        """
        모든 track 상태를 초기화한다.
        """

        self.states.clear()
        
class RiskStabilizerV2:
    """
    Risk Level 안정화 V2.

    원칙:
    - 위험 수준 상승은 즉시 반영한다.
    - 현재 안정화 상태보다 낮은 위험이 연속으로 관찰되면
      release_frames 이후 위험 수준을 낮춘다.
    - 하락 대기 중 SAFE/CAUTION이 섞여도
      현재 stable level보다 낮기만 하면 카운트를 유지한다.
    - 다시 현재 stable level 이상이 나오면
      하락 카운트를 초기화한다.
    """

    def __init__(self, release_frames=3):

        if release_frames < 1:
            raise ValueError(
                "release_frames는 1 이상이어야 합니다."
            )

        self.release_frames = release_frames
        self.states = {}

    def update(
        self,
        track_id,
        raw_risk_level,
    ):

        if raw_risk_level not in RISK_PRIORITY:
            raise ValueError(
                f"알 수 없는 risk level: {raw_risk_level}"
            )

        # 처음 등장한 객체
        if track_id not in self.states:

            self.states[track_id] = {
                "stable_level": raw_risk_level,
                "lower_count": 0,
                "lowest_level": raw_risk_level,
            }

            return raw_risk_level

        state = self.states[
            track_id
        ]

        stable_level = state[
            "stable_level"
        ]

        stable_priority = RISK_PRIORITY[
            stable_level
        ]

        raw_priority = RISK_PRIORITY[
            raw_risk_level
        ]

        # --------------------------------------------------
        # 같은 상태
        # --------------------------------------------------

        if raw_priority == stable_priority:

            state["lower_count"] = 0
            state["lowest_level"] = stable_level

            return stable_level

        # --------------------------------------------------
        # 위험 상승
        # 즉시 반영
        # --------------------------------------------------

        if raw_priority > stable_priority:

            state["stable_level"] = raw_risk_level
            state["lower_count"] = 0
            state["lowest_level"] = raw_risk_level

            return raw_risk_level

        # --------------------------------------------------
        # 위험 하락 후보
        #
        # 정확히 같은 상태일 필요 없이
        # 현재 stable보다 낮으면 카운트
        # --------------------------------------------------

        state["lower_count"] += 1

        lowest_level = state[
            "lowest_level"
        ]

        if (
            RISK_PRIORITY[raw_risk_level]
            < RISK_PRIORITY[lowest_level]
        ):

            state[
                "lowest_level"
            ] = raw_risk_level

        # --------------------------------------------------
        # 하락 확정
        # --------------------------------------------------

        if (
            state["lower_count"]
            >= self.release_frames
        ):

            new_level = state[
                "lowest_level"
            ]

            state[
                "stable_level"
            ] = new_level

            state[
                "lower_count"
            ] = 0

            state[
                "lowest_level"
            ] = new_level

        return state[
            "stable_level"
        ]

    def remove_track(
        self,
        track_id,
    ):

        self.states.pop(
            track_id,
            None,
        )

    def reset(self):

        self.states.clear()