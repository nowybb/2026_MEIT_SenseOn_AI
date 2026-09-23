"""실시간 SenseOn AI1/AI2 프레임 처리기.

YOLO + BoT-SORT 객체 추적 결과를 기반으로
AI2의 접근 여부, 안정화 궤적, Collision Zone, Visual TTC를 계산한다.

최종 위험도 판단에는
Stable Trajectory + TTC Required 방식을 사용한다.

- BLUE: 기존 baseline 1초 궤적 (비교용)
- MAGENTA: 최종 위험도 판단에 사용하는 stable 0.3초 궤적
"""

import cv2

from ai2.track_history import TrackHistory
from ai2.approach import is_approaching
from ai2.trajectory import (
    predict_future_position,
    predict_stable_position,
)
from ai2.collision import (
    create_collision_zone,
    check_path_collision,
)
from ai2.ttc import calculate_visual_ttc
from ai2.risk import (
    determine_risk_level,
    select_primary_hazard,
)
from senseon_protocol import build_final_result


# ============================================================
# 설정
# ============================================================

# COCO class:
# bicycle=1, car=2, motorcycle=3, bus=5, truck=7
TARGET_CLASSES = [1, 2, 3, 5, 7]

HISTORY_SIZE = 10

# 기존 baseline 궤적
BASELINE_FUTURE_TIME = 1.0

# AI2 최종 적용 stable 궤적
STABLE_FUTURE_TIME = 0.3

COLORS = {
    "SAFE": (0, 255, 0),
    "CAUTION": (0, 255, 255),
    "DANGER": (0, 0, 255),
    "UNKNOWN": (200, 200, 200),
}


# ============================================================
# Frame Analyzer
# ============================================================

class FrameAnalyzer:

    def __init__(
        self,
        model_path,
        mirror_direction=False,
    ):
        from ultralytics import YOLO

        # YOLO 모델 로드
        self.model = YOLO(
            str(model_path)
        )

        # track별 최근 기록 저장
        self.history = TrackHistory(
            max_history=HISTORY_SIZE
        )

        # 카메라 좌우 반전 여부
        self.mirror_direction = (
            mirror_direction
        )

        # 같은 track ID가 오랜 시간 뒤 다시 등장했을 때
        # 이전 기록과 연결되는 문제 방지
        self.last_time_by_id = {}


    # ========================================================
    # 프레임 처리
    # ========================================================

    def process(
        self,
        frame,
        timestamp,
        annotate=False,
    ):
        """
        반환:
            final_result, state, annotated_frame

        state='READY'
            분석 완료 상태.

            final_result가 None이면
            현재 경고할 위험 객체가 없음을 의미함.

        state='UNKNOWN'
            객체는 탐지됐지만
            track ID 또는 history가 충분하지 않아
            아직 SAFE라고 단정할 수 없는 상태.
        """

        height, width = frame.shape[:2]

        # 사용자 후방 위험 영역
        collision_zone = (
            create_collision_zone(
                width,
                height,
            )
        )

        # ====================================================
        # 1. YOLO + BoT-SORT
        # ====================================================

        result = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=TARGET_CLASSES,
            conf=0.3,
            verbose=False,
        )[0]

        boxes = result.boxes

        # 현재 프레임의 위험 객체 후보
        hazards = []

        # 화면 시각화용 데이터
        drawings = []

        # 탐지는 됐지만 아직 판단 불가능한 객체 존재 여부
        unknown_present = False


        # ====================================================
        # Track ID가 아직 생성되지 않은 탐지 객체
        # ====================================================

        if (
            boxes is not None
            and len(boxes) > 0
            and boxes.id is None
        ):
            unknown_present = True


        # ====================================================
        # Track ID가 있는 객체 처리
        # ====================================================

        if (
            boxes is not None
            and boxes.id is not None
        ):

            for box in boxes:

                # --------------------------------------------
                # 객체 정보
                # --------------------------------------------

                track_id = int(
                    box.id[0]
                )

                class_id = int(
                    box.cls[0]
                )

                class_name = (
                    self.model.names[
                        class_id
                    ]
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = map(
                    float,
                    box.xyxy[0]
                )

                center_x = (
                    x1 + x2
                ) / 2.0

                center_y = (
                    y1 + y2
                ) / 2.0

                bbox = [
                    x1,
                    y1,
                    x2,
                    y2,
                ]

                current_position = (
                    center_x,
                    center_y,
                )


                # --------------------------------------------
                # Track History에 저장할 데이터
                # --------------------------------------------

                detection = {
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "bbox": bbox,
                    "center_x": center_x,
                    "center_y": center_y,
                }


                # --------------------------------------------
                # 같은 ID가 긴 시간 뒤 재등장한 경우
                # 이전 history 제거
                # --------------------------------------------

                last_seen = (
                    self.last_time_by_id.get(
                        track_id
                    )
                )

                if (
                    last_seen is not None
                    and timestamp - last_seen
                    > 0.75
                ):
                    self.history.history[
                        track_id
                    ].clear()

                self.last_time_by_id[
                    track_id
                ] = timestamp


                # --------------------------------------------
                # Track History 업데이트
                # --------------------------------------------

                self.history.update(
                    detection
                )

                history = (
                    self.history.get_history(
                        track_id
                    )
                )


                # --------------------------------------------
                # AI2 계산 가능한 최소 history 확인
                # --------------------------------------------

                enough_history = (
                    len(history) >= 3
                    and (
                        history[-1]["timestamp"]
                        - history[0]["timestamp"]
                    ) >= 0.1
                )


                # 초기값
                risk = "UNKNOWN"

                baseline_position = None
                stable_position = None

                baseline_path_collision = False
                stable_path_collision = False

                visual_ttc = None


                # =================================================
                # AI2 위험 분석
                # =================================================

                if enough_history:

                    # =============================================
                    # 2. 접근 여부
                    # =============================================

                    approaching = (
                        is_approaching(
                            history
                        )
                    )


                    # =============================================
                    # 3. 기존 Baseline Trajectory
                    #
                    # 1초 후 위치 예측
                    # 비교/시각화용
                    # =============================================

                    baseline_position = (
                        predict_future_position(
                            history,
                            future_time=(
                                BASELINE_FUTURE_TIME
                            ),
                        )
                    )

                    if (
                        baseline_position
                        is not None
                    ):
                        baseline_path_collision = (
                            check_path_collision(
                                current_position,
                                baseline_position,
                                collision_zone,
                            )
                        )


                    # =============================================
                    # 4. Stable Trajectory
                    #
                    # AI2 최종 위험도 판단에 사용
                    # =============================================

                    stable_position = (
                        predict_stable_position(
                            history,
                            future_time=(
                                STABLE_FUTURE_TIME
                            ),
                            window_size=(
                                HISTORY_SIZE
                            ),
                        )
                    )

                    if (
                        stable_position
                        is not None
                    ):
                        stable_path_collision = (
                            check_path_collision(
                                current_position,
                                stable_position,
                                collision_zone,
                            )
                        )


                    # =============================================
                    # 5. Visual TTC
                    # =============================================

                    visual_ttc = (
                        calculate_visual_ttc(
                            history
                        )
                    )


                    # =============================================
                    # 6. 최종 Risk
                    #
                    # AI2 Ground Truth 평가 후 적용:
                    #
                    # Stable Trajectory
                    # +
                    # TTC Required
                    #
                    # TTC 계산 불가능 시 SAFE
                    # =============================================

                    if visual_ttc is None:

                        risk = "SAFE"

                    else:

                        risk = (
                            determine_risk_level(
                                approaching=(
                                    approaching
                                ),
                                path_collision=(
                                    stable_path_collision
                                ),
                                visual_ttc=(
                                    visual_ttc
                                ),
                            )
                        )


                    # =============================================
                    # 7. Primary Hazard 후보 추가
                    # =============================================

                    hazards.append({
                        "track_id": track_id,
                        "class_name": class_name,
                        "center_x": center_x,
                        "approaching": approaching,
                        "path_collision": (
                            stable_path_collision
                        ),
                        "visual_ttc": visual_ttc,
                        "risk_level": risk,
                    })


                # history 부족
                else:

                    unknown_present = True


                # =================================================
                # 화면 시각화 정보 저장
                # =================================================

                drawings.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "bbox": bbox,
                    "center": current_position,

                    "baseline_position": (
                        baseline_position
                    ),

                    "stable_position": (
                        stable_position
                    ),

                    "baseline_path_collision": (
                        baseline_path_collision
                    ),

                    "stable_path_collision": (
                        stable_path_collision
                    ),

                    "risk": risk,
                })


        # ====================================================
        # 8. Primary Hazard 선정
        # ====================================================

        primary = (
            select_primary_hazard(
                hazards
            )
        )


        # ====================================================
        # 9. 최종 통신 결과 생성
        # ====================================================

        if primary is not None:

            final_result = (
                build_final_result(
                    primary,
                    width,
                    mirror=(
                        self.mirror_direction
                    ),
                )
            )

            state = "READY"


        # 분석 불가능 객체 존재
        elif unknown_present:

            final_result = None
            state = "UNKNOWN"


        # 모든 객체 분석 완료 + 위험 객체 없음
        else:

            final_result = None
            state = "READY"


        # ====================================================
        # annotate 필요 없는 경우
        # ====================================================

        if not annotate:

            return (
                final_result,
                state,
                frame,
            )


        # ====================================================
        # 10. 화면 시각화
        # ====================================================

        out = frame.copy()


        # Collision Zone
        zx1, zy1, zx2, zy2 = (
            collision_zone
        )

        cv2.rectangle(
            out,
            (
                int(zx1),
                int(zy1),
            ),
            (
                int(zx2),
                int(zy2),
            ),
            (255, 255, 0),
            2,
        )


        # ====================================================
        # 객체별 시각화
        # ====================================================

        for drawing in drawings:

            track_id = drawing[
                "track_id"
            ]

            class_name = drawing[
                "class_name"
            ]

            bbox = drawing[
                "bbox"
            ]

            center = drawing[
                "center"
            ]

            baseline_position = drawing[
                "baseline_position"
            ]

            stable_position = drawing[
                "stable_position"
            ]

            risk = drawing[
                "risk"
            ]

            x1, y1, x2, y2 = bbox


            # ================================================
            # BLUE:
            # 기존 baseline 궤적
            # ================================================

            if (
                baseline_position
                is not None
            ):
                cv2.arrowedLine(
                    out,
                    tuple(
                        map(
                            int,
                            center,
                        )
                    ),
                    tuple(
                        map(
                            int,
                            baseline_position,
                        )
                    ),
                    (255, 0, 0),
                    2,
                    tipLength=0.15,
                )


            # ================================================
            # MAGENTA:
            # 최종 위험 판단에 사용되는 stable 궤적
            # ================================================

            if (
                stable_position
                is not None
            ):
                cv2.arrowedLine(
                    out,
                    tuple(
                        map(
                            int,
                            center,
                        )
                    ),
                    tuple(
                        map(
                            int,
                            stable_position,
                        )
                    ),
                    (255, 0, 255),
                    2,
                    tipLength=0.15,
                )


            # ================================================
            # Bounding Box
            # ================================================

            is_primary = (
                primary is not None
                and track_id
                == primary["track_id"]
            )

            thickness = (
                3
                if is_primary
                else 2
            )

            cv2.rectangle(
                out,
                (
                    int(x1),
                    int(y1),
                ),
                (
                    int(x2),
                    int(y2),
                ),
                COLORS[risk],
                thickness,
            )


            # ================================================
            # 객체 정보
            # ================================================

            label = (
                f"ID:{track_id} "
                f"{class_name} "
                f"{risk}"
            )

            cv2.putText(
                out,
                label,
                (
                    max(
                        0,
                        int(x1),
                    ),
                    max(
                        95,
                        int(y1) - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[risk],
                2,
            )


        # ====================================================
        # 11. 상단 Primary Hazard 정보
        # ====================================================

        if final_result is not None:

            text = (
                f"HAZARD: "
                f"{final_result['object']} "
                f"{final_result['direction']} "
                f"{final_result['risk']}"
            )

        elif state == "UNKNOWN":

            text = (
                "ANALYSIS PENDING"
            )

        else:

            text = (
                "HAZARD: NONE"
            )


        # 상단 배경
        cv2.rectangle(
            out,
            (0, 0),
            (width, 70),
            (30, 30, 30),
            -1,
        )


        # 위험 객체 정보
        cv2.putText(
            out,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


        # 궤적 설명
        cv2.putText(
            out,
            (
                "BLUE: baseline / "
                "MAGENTA: final stable trajectory"
            ),
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )


        return (
            final_result,
            state,
            out,
        )