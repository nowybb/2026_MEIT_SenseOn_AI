# main_video.py

from pathlib import Path
import csv
import math

import cv2
from ultralytics import YOLO

# ============================================================
# AI2 모듈
# ============================================================

from ai2.track_history import TrackHistory

from ai2.approach import (
    calculate_bbox_area,
    calculate_approach_rate,
    is_approaching,
)

from ai2.trajectory import predict_future_position

from ai2.collision import (
    create_collision_zone,
    check_path_collision,
    calculate_bbox_zone_overlap,
    check_bbox_path_collision,
)

from ai2.proximity import calculate_proximity_features

from ai2.ttc import calculate_visual_ttc

from ai2.risk import (
    determine_risk_level,
    select_primary_hazard,
)


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI1_DIR = BASE_DIR / "ai1"

VIDEO_DIR = AI1_DIR / "videos"
OUTPUT_DIR = AI1_DIR / "outputs"

MODEL_PATH = AI1_DIR / "yolo11n.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 전체 테스트 영상 6개
VIDEO_NAMES = [
    "test1.mp4",
    "test2.mp4",
    "test3.mp4",
    "test4.mp4",
    "test5.mp4",
    "test_receding.mp4",
]


HISTORY_SIZE = 10

FUTURE_TIME = 1.0

CONFIDENCE = 0.3

# COCO 클래스:
# 1 = bicycle, 2 = car, 3 = motorcycle
TARGET_CLASSES = [1, 2, 3]


# 위험도 변화 상세 출력 대상
DEBUG_VIDEO = "test_receding.mp4"


# 위험도별 색상 (OpenCV: BGR)
RISK_COLORS = {
    "SAFE": (0, 255, 0),
    "CAUTION": (0, 255, 255),
    "DANGER": (0, 0, 255),
    "UNKNOWN": (200, 200, 200),
}


# ============================================================
# 2. 장기 접근 추세 계산
#
# 디버깅 전용
# 실제 위험 판단에는 사용하지 않음
# ============================================================

def calculate_long_approach_rate(history):

    if len(history) < 2:
        return 0.0

    first = history[0]
    last = history[-1]

    first_area = calculate_bbox_area(first["bbox"])
    last_area = calculate_bbox_area(last["bbox"])

    delta_time = last["timestamp"] - first["timestamp"]

    if first_area <= 0 or delta_time <= 0:
        return 0.0

    area_change_ratio = (
        last_area - first_area
    ) / first_area

    return area_change_ratio / delta_time


# ============================================================
# 3. 위험도 변화 디버깅 출력
# ============================================================

def print_risk_change(
    frame_index,
    timestamp,
    track_id,
    class_name,
    previous_risk,
    risk_level,
    approach_rate,
    long_approach_rate,
    approaching,
    path_collision,
    bbox_path_collision,
    bbox_zone_overlap,
    proximity,
    visual_ttc,
    current_area,
    history_length,
):

    print("\n-----------------------------------")
    print("[RISK CHANGE]")

    print("Frame:", frame_index)
    print("Time:", round(timestamp, 3))

    print("Track ID:", track_id)
    print("Class:", class_name)

    print(
        "Risk:",
        previous_risk,
        "->",
        risk_level
    )

    print(
        "ApproachRate:",
        round(approach_rate, 4)
    )

    print(
        "LongApproachRate:",
        round(long_approach_rate, 4)
    )

    print("Approaching:", approaching)

    # 기존 중심점 기반 충돌 판단
    print(
        "PathCollision:",
        path_collision
    )

    # 신규 bbox 기반 충돌 판단
    print(
        "BBoxPathCollision:",
        bbox_path_collision
    )

    # 현재 bbox가 Zone과 겹치는 비율
    print(
        "BBoxZoneOverlap:",
        round(bbox_zone_overlap, 4)
    )

    # 상대적 근접도 특징
    print(
        "HeightRatio:",
        round(proximity["height_ratio"], 4)
    )

    print(
        "AreaRatio:",
        round(proximity["area_ratio"], 4)
    )

    print(
        "BottomYRatio:",
        round(proximity["bottom_y_ratio"], 4)
    )

    print("Visual TTC:", visual_ttc)

    print(
        "BBox Area:",
        round(current_area, 2)
    )

    print("History Length:", history_length)

    print("-----------------------------------")


# ============================================================
# 4. 영상 하나 분석
# ============================================================

def process_video(video_name):

    video_path = VIDEO_DIR / video_name

    output_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_risk_result.mp4"
    )

    csv_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_risk_log.csv"
    )

    print("\n===================================")
    print("영상 분석 시작:", video_name)
    print("===================================")


    # ========================================================
    # 5. 파일 확인
    # ========================================================

    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}"
        )


    # ========================================================
    # 6. 영상 정보 확인
    # ========================================================

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()

        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    finally:
        cap.release()


    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            "영상 FPS를 확인할 수 없습니다."
        )

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            "영상 크기를 확인할 수 없습니다."
        )

    print("FPS:", fps)

    print(
        "Resolution:",
        frame_width,
        "x",
        frame_height
    )


    # ========================================================
    # 7. 결과 영상 저장 설정
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        writer.release()

        raise RuntimeError(
            f"결과 영상을 생성할 수 없습니다: {output_path}"
        )


    frame_index = 0


    # ========================================================
    # 8. 영상 분석
    # ========================================================

    try:

        # CSV 파일 생성
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            # =================================================
            # 9. CSV 헤더
            # =================================================

            csv_writer.writerow([
                "frame",
                "timestamp",

                "track_id",
                "class_name",

                "bbox_area",

                "approach_rate",
                "long_approach_rate",
                "approaching",

                # 기존 중심점 방식
                "path_collision",

                # 신규 bbox 방식
                "bbox_path_collision",
                "bbox_zone_overlap",

                # Proximity 특징
                "height_ratio",
                "area_ratio",
                "bottom_y_ratio",

                "visual_ttc",
                "risk_level",

                "center_x",
                "center_y",

                "predicted_x",
                "predicted_y",
            ])


            # =================================================
            # 10. YOLO 모델
            # 영상마다 새 모델 생성
            # =================================================

            model = YOLO(str(MODEL_PATH))


            # =================================================
            # 11. Track History
            # =================================================

            track_history = TrackHistory(
                max_history=HISTORY_SIZE
            )

            last_risk_by_id = {}


            # =================================================
            # 12. Collision Zone
            # =================================================

            collision_zone = create_collision_zone(
                frame_width,
                frame_height
            )

            print(
                "Collision Zone:",
                collision_zone
            )


            # =================================================
            # 13. YOLO + BoT-SORT
            # =================================================

            results = model.track(
                source=str(video_path),
                tracker="botsort.yaml",
                classes=TARGET_CLASSES,
                conf=CONFIDENCE,
                persist=True,
                stream=True,
                verbose=False,
            )


            # =================================================
            # 14. 프레임별 처리
            # =================================================

            for result in results:

                frame_index += 1

                frame = result.orig_img.copy()

                # 원본 FPS 기준 근사 시간
                timestamp = (frame_index - 1) / fps

                boxes = result.boxes

                hazards = []
                display_info = []


                # =============================================
                # 15. 객체별 분석
                # =============================================

                if (
                    boxes is not None
                    and boxes.id is not None
                ):

                    for box in boxes:

                        # -------------------------------------
                        # AI1 객체 정보
                        # -------------------------------------

                        track_id = int(box.id[0])
                        class_id = int(box.cls[0])

                        class_name = model.names[class_id]

                        confidence = float(box.conf[0])

                        x1, y1, x2, y2 = map(
                            float,
                            box.xyxy[0]
                        )

                        bbox = [x1, y1, x2, y2]

                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        current_position = (
                            center_x,
                            center_y
                        )


                        # -------------------------------------
                        # AI1 → AI2 입력 데이터
                        # -------------------------------------

                        detection = {
                            "track_id": track_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "timestamp": timestamp,
                            "bbox": bbox,
                            "center_x": center_x,
                            "center_y": center_y,
                        }


                        # -------------------------------------
                        # Track History 업데이트
                        # -------------------------------------

                        track_history.update(detection)

                        history = track_history.get_history(
                            track_id
                        )


                        # =====================================
                        # 16. bbox 면적
                        # =====================================

                        current_area = calculate_bbox_area(
                            bbox
                        )


                        # =====================================
                        # 17. Proximity 특징 계산
                        #
                        # 모든 탐지 객체에 대해 계산
                        # 아직 위험도 판단에는 사용하지 않음
                        # =====================================

                        proximity = calculate_proximity_features(
                            bbox=bbox,
                            frame_width=frame_width,
                            frame_height=frame_height
                        )

                        height_ratio = proximity[
                            "height_ratio"
                        ]

                        area_ratio = proximity[
                            "area_ratio"
                        ]

                        bottom_y_ratio = proximity[
                            "bottom_y_ratio"
                        ]


                        # =====================================
                        # 18. 현재 bbox와 Zone 겹침 비율
                        # =====================================

                        bbox_zone_overlap = (
                            calculate_bbox_zone_overlap(
                                bbox,
                                collision_zone
                            )
                        )


                        # =====================================
                        # 19. AI2 초기값
                        # =====================================

                        approach_rate = 0.0
                        long_approach_rate = 0.0

                        approaching = False

                        # 기존 방식
                        path_collision = False

                        # 신규 방식
                        # 분석 기록이 부족하면 계산하지 않음
                        bbox_path_collision = None

                        visual_ttc = None
                        predicted_position = None

                        risk_level = "UNKNOWN"


                        # =====================================
                        # 20. AI2 위험 판단
                        # =====================================

                        enough_history = (
                            len(history) >= 3
                            and (
                                history[-1]["timestamp"]
                                - history[0]["timestamp"]
                            ) >= 0.1
                        )

                        if enough_history:

                            # ---------------------------------
                            # ApproachRate
                            # ---------------------------------

                            approach_rate = calculate_approach_rate(
                                history
                            )

                            approaching = is_approaching(
                                history
                            )

                            # ---------------------------------
                            # 장기 접근 추세
                            # 디버깅 전용
                            # ---------------------------------

                            long_approach_rate = (
                                calculate_long_approach_rate(
                                    history
                                )
                            )

                            # ---------------------------------
                            # Trajectory
                            # ---------------------------------

                            predicted_position = (
                                predict_future_position(
                                    history,
                                    future_time=FUTURE_TIME
                                )
                            )

                            # ---------------------------------
                            # 기존 중심점 기반 충돌 판단
                            # ---------------------------------

                            path_collision = check_path_collision(
                                current_position,
                                predicted_position,
                                collision_zone
                            )

                            # ---------------------------------
                            # 신규 bbox 기반 충돌 판단
                            #
                            # 비교용으로만 사용
                            # ---------------------------------

                            bbox_path_collision = (
                                check_bbox_path_collision(
                                    bbox=bbox,
                                    current_position=current_position,
                                    predicted_position=predicted_position,
                                    collision_zone=collision_zone
                                )
                            )

                            # ---------------------------------
                            # Visual TTC
                            # ---------------------------------

                            visual_ttc = calculate_visual_ttc(
                                history
                            )

                            # ---------------------------------
                            # Risk
                            #
                            # 주의:
                            # 아직 기존 path_collision 사용
                            # bbox_path_collision 사용 X
                            # proximity 사용 X
                            # ---------------------------------

                            risk_level = determine_risk_level(
                                approaching,
                                path_collision,
                                visual_ttc
                            )

                            # ---------------------------------
                            # 현재 프레임 위험 객체 저장
                            # ---------------------------------

                            hazards.append({
                                "track_id": track_id,
                                "class_name": class_name,
                                "approach_rate": approach_rate,
                                "approaching": approaching,
                                "path_collision": path_collision,
                                "visual_ttc": visual_ttc,
                                "risk_level": risk_level,
                            })


                        # =====================================
                        # 21. 위험도 변화 디버깅
                        # =====================================

                        if video_name == DEBUG_VIDEO:

                            previous_risk = last_risk_by_id.get(
                                track_id,
                                "UNKNOWN"
                            )

                            if previous_risk != risk_level:

                                if (
                                    risk_level in ("CAUTION", "DANGER")
                                    or previous_risk in ("CAUTION", "DANGER")
                                ):

                                    print_risk_change(
                                        frame_index=frame_index,
                                        timestamp=timestamp,
                                        track_id=track_id,
                                        class_name=class_name,
                                        previous_risk=previous_risk,
                                        risk_level=risk_level,
                                        approach_rate=approach_rate,
                                        long_approach_rate=long_approach_rate,
                                        approaching=approaching,
                                        path_collision=path_collision,
                                        bbox_path_collision=bbox_path_collision,
                                        bbox_zone_overlap=bbox_zone_overlap,
                                        proximity=proximity,
                                        visual_ttc=visual_ttc,
                                        current_area=current_area,
                                        history_length=len(history),
                                    )

                            last_risk_by_id[track_id] = (
                                risk_level
                            )


                        # =====================================
                        # 22. CSV 기록
                        # =====================================

                        if predicted_position is None:
                            predicted_x = ""
                            predicted_y = ""

                        else:
                            predicted_x = predicted_position[0]
                            predicted_y = predicted_position[1]


                        csv_writer.writerow([
                            frame_index,
                            timestamp,

                            track_id,
                            class_name,

                            current_area,

                            approach_rate,
                            long_approach_rate,
                            approaching,

                            # 기존 중심점 방식
                            path_collision,

                            # 신규 bbox 방식
                            (
                                bbox_path_collision
                                if bbox_path_collision is not None
                                else ""
                            ),

                            bbox_zone_overlap,

                            # 근접도 특징
                            height_ratio,
                            area_ratio,
                            bottom_y_ratio,

                            visual_ttc,
                            risk_level,

                            center_x,
                            center_y,

                            predicted_x,
                            predicted_y,
                        ])


                        # =====================================
                        # 23. 영상 시각화 정보
                        # =====================================

                        display_info.append({
                            "track_id": track_id,
                            "class_name": class_name,
                            "bbox": bbox,

                            "center_x": center_x,
                            "center_y": center_y,

                            "predicted_position": predicted_position,

                            "risk_level": risk_level,
                        })


                # =============================================
                # 24. Primary Hazard
                # =============================================

                primary_hazard = select_primary_hazard(
                    hazards
                )


                # =============================================
                # 25. Collision Zone 표시
                # =============================================

                zx1, zy1, zx2, zy2 = collision_zone

                cv2.rectangle(
                    frame,
                    (int(zx1), int(zy1)),
                    (int(zx2), int(zy2)),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Collision Zone (Proxy)",
                    (
                        int(zx1),
                        min(
                            frame_height - 10,
                            int(zy1) + 20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )


                # =============================================
                # 26. 객체별 시각화
                # =============================================

                for info in display_info:

                    x1, y1, x2, y2 = info["bbox"]

                    track_id = info["track_id"]
                    risk_level = info["risk_level"]

                    color = RISK_COLORS[risk_level]


                    # -----------------------------------------
                    # 예측 이동경로
                    # -----------------------------------------

                    predicted = info[
                        "predicted_position"
                    ]

                    if predicted is not None:

                        start_point = (
                            int(info["center_x"]),
                            int(info["center_y"])
                        )

                        end_point = (
                            int(predicted[0]),
                            int(predicted[1])
                        )

                        # 파란색 예측 경로
                        cv2.arrowedLine(
                            frame,
                            start_point,
                            end_point,
                            (255, 0, 0),
                            2,
                            tipLength=0.15
                        )

                        # 예상 도착점
                        if (
                            0 <= end_point[0] < frame_width
                            and 0 <= end_point[1] < frame_height
                        ):

                            cv2.circle(
                                frame,
                                end_point,
                                6,
                                (255, 0, 0),
                                -1
                            )


                    # -----------------------------------------
                    # 대표 위험 객체 강조
                    # -----------------------------------------

                    thickness = 2

                    if (
                        primary_hazard is not None
                        and track_id
                        == primary_hazard["track_id"]
                    ):
                        thickness = 4


                    # -----------------------------------------
                    # Bounding Box
                    # -----------------------------------------

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        thickness
                    )


                    # -----------------------------------------
                    # Label
                    # -----------------------------------------

                    label = (
                        f"ID:{track_id} "
                        f"{info['class_name']} "
                        f"{risk_level}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            max(0, int(x1)),
                            max(90, int(y1) - 10)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )


                # =============================================
                # 27. 상단 정보 표시
                # =============================================

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame_width, 75),
                    (30, 30, 30),
                    -1
                )

                cv2.putText(
                    frame,
                    f"Frame: {frame_index}",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


                # ---------------------------------------------
                # Primary Hazard 표시
                # ---------------------------------------------

                if primary_hazard is None:

                    primary_text = "Primary Hazard: NONE"

                    primary_color = (0, 255, 0)

                else:

                    primary_text = (
                        "Primary Hazard: "
                        f"ID {primary_hazard['track_id']} "
                        f"{primary_hazard['risk_level']}"
                    )

                    primary_color = RISK_COLORS[
                        primary_hazard["risk_level"]
                    ]


                cv2.putText(
                    frame,
                    primary_text,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    primary_color,
                    2
                )


                # =============================================
                # 28. 결과 영상 저장
                # =============================================

                writer.write(frame)


    # ========================================================
    # 29. 영상 및 CSV 종료
    # ========================================================

    finally:
        writer.release()


    if frame_index == 0:
        raise RuntimeError(
            "처리된 프레임이 없습니다."
        )


    print("\n===================================")
    print("영상 분석 완료:", video_name)

    print("처리 프레임:", frame_index)

    print("결과 영상:", output_path)
    print("CSV 로그:", csv_path)

    print("===================================")

    return output_path


# ============================================================
# 30. 전체 영상 순차 실행
# ============================================================

success = []
failed = []

for video_name in VIDEO_NAMES:

    try:
        process_video(video_name)

        success.append(video_name)

    except Exception as e:

        print(
            "분석 실패:",
            video_name
        )

        print(
            "오류:",
            e
        )

        failed.append(video_name)


# ============================================================
# 31. 최종 결과
# ============================================================

print("\n===================================")
print("AI1 + AI2 통합 테스트 종료")
print("===================================")

print("성공:", len(success))
print("실패:", len(failed))

print("\n성공 영상:")

for name in success:
    print("-", name)

print("\n실패 영상:")

for name in failed:
    print("-", name)

print("===================================")

if failed:
    raise SystemExit(1)