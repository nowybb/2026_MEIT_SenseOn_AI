from pathlib import Path
from ultralytics import YOLO
import cv2
import csv
import math

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
)

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


# ------------------------------------------------------------
# 이번에는 멀어지는 영상만 디버깅
# ------------------------------------------------------------

VIDEO_NAMES = [
    "test1.mp4",
    "test2.mp4",
    "test3.mp4",
    "test4.mp4",
    "test5.mp4",
    "test_receding.mp4"
]



HISTORY_SIZE = 10

FUTURE_TIME = 1.0

CONFIDENCE = 0.3

TARGET_CLASSES = [1, 2, 3]

# 디버깅 대상 영상
DEBUG_VIDEO = "test_receding.mp4"


# ============================================================
# 2. 위험도별 색상
# ============================================================

RISK_COLORS = {

    "SAFE": (0, 255, 0),

    "CAUTION": (0, 255, 255),

    "DANGER": (0, 0, 255),

    "UNKNOWN": (200, 200, 200),

}


# ============================================================
# 3. 장기 접근 추세 계산
#
# AI2 접근 판단에는 사용하지 않음.
# 디버깅 목적으로만 사용.
# ============================================================

def calculate_long_approach_rate(history):

    if len(history) < 2:
        return 0.0

    old = history[0]
    current = history[-1]

    old_area = calculate_bbox_area(
        old["bbox"]
    )

    current_area = calculate_bbox_area(
        current["bbox"]
    )

    delta_time = (
        current["timestamp"]
        - old["timestamp"]
    )

    if old_area <= 0 or delta_time <= 0:
        return 0.0

    area_change_ratio = (
        current_area - old_area
    ) / old_area

    return area_change_ratio / delta_time


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

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

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
    # 7. 결과 영상 저장
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            "결과 영상 생성 실패"
        )


    # ========================================================
    # 8. CSV 저장 설정
    # ========================================================

    try:

        csv_file = open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        )

    except Exception:

        writer.release()

        raise


    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow([

        "frame",
        "timestamp",

        "track_id",
        "class_name",

        "bbox_area",
        "approach_rate",
        "long_approach_rate",

        "approaching",
        "path_collision",

        "visual_ttc",
        "risk_level",

        "center_x",
        "center_y",

        "predicted_x",
        "predicted_y"

    ])


    frame_index = 0


    # ========================================================
    # 9. 분석 시작
    # ========================================================

    try:

        # ----------------------------------------------------
        # YOLO 모델
        # 영상마다 새 모델 생성 → 추적 상태 분리
        # ----------------------------------------------------

        model = YOLO(
            str(MODEL_PATH)
        )


        # ----------------------------------------------------
        # AI2 Track History
        # ----------------------------------------------------

        track_history = TrackHistory(
            max_history=HISTORY_SIZE
        )


        # ----------------------------------------------------
        # 위험도 변화 기록
        # ----------------------------------------------------

        last_risk_by_id = {}


        # ----------------------------------------------------
        # Collision Zone
        # ----------------------------------------------------

        collision_zone = create_collision_zone(
            frame_width,
            frame_height
        )

        print(
            "Collision Zone:",
            collision_zone
        )


        # ----------------------------------------------------
        # YOLO + BoT-SORT
        # ----------------------------------------------------

        results = model.track(

            source=str(video_path),

            tracker="botsort.yaml",

            classes=TARGET_CLASSES,

            conf=CONFIDENCE,

            persist=True,

            stream=True,

            verbose=False

        )


        # ====================================================
        # 10. 프레임별 처리
        # ====================================================

        for result in results:

            frame_index += 1

            frame = result.orig_img.copy()

            # 영상 FPS를 기준으로 한 근사 시간
            timestamp = (
                frame_index - 1
            ) / fps

            boxes = result.boxes

            hazards = []

            display_info = []


            # =================================================
            # 11. 객체별 분석
            # =================================================

            if (
                boxes is not None
                and boxes.id is not None
            ):

                for box in boxes:

                    # ----------------------------------------
                    # AI1: 객체 정보
                    # ----------------------------------------

                    track_id = int(
                        box.id[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = model.names[
                        class_id
                    ]

                    confidence = float(
                        box.conf[0]
                    )

                    x1, y1, x2, y2 = map(
                        float,
                        box.xyxy[0]
                    )

                    center_x = (
                        x1 + x2
                    ) / 2

                    center_y = (
                        y1 + y2
                    ) / 2


                    # ----------------------------------------
                    # AI1 → AI2 입력 데이터
                    # ----------------------------------------

                    detection = {

                        "track_id": track_id,

                        "class_name": class_name,

                        "confidence": confidence,

                        "timestamp": timestamp,

                        "bbox": [
                            x1, y1, x2, y2
                        ],

                        "center_x": center_x,

                        "center_y": center_y

                    }


                    # ----------------------------------------
                    # History 업데이트
                    # ----------------------------------------

                    track_history.update(
                        detection
                    )

                    history = (
                        track_history.get_history(
                            track_id
                        )
                    )


                    # ----------------------------------------
                    # 현재 bbox 면적
                    # ----------------------------------------

                    current_area = calculate_bbox_area(
                        detection["bbox"]
                    )


                    # ----------------------------------------
                    # 초기 상태
                    # ----------------------------------------

                    approach_rate = 0.0

                    long_approach_rate = 0.0

                    approaching = False

                    path_collision = False

                    visual_ttc = None

                    predicted_position = None

                    risk_level = "UNKNOWN"


                    # =================================================
                    # 12. AI2 위험 판단
                    # =================================================

                    if (
                        len(history) >= 3
                        and history[-1]["timestamp"] - history[0]["timestamp"] >= 0.1
                        ):


                        # ------------------------------------
                        # Approach Rate
                        # ------------------------------------

                        approach_rate = (
                            calculate_approach_rate(
                                history
                            )
                        )

                        approaching = (
                            is_approaching(
                                history
                            )
                        )


                        # ------------------------------------
                        # 장기 접근 추세
                        # 디버깅 전용
                        # ------------------------------------

                        long_approach_rate = (
                            calculate_long_approach_rate(
                                history
                            )
                        )


                        # ------------------------------------
                        # Trajectory
                        # ------------------------------------

                        current_position = (
                            center_x,
                            center_y
                        )

                        predicted_position = (
                            predict_future_position(

                                history,

                                future_time=FUTURE_TIME

                            )
                        )


                        # ------------------------------------
                        # Collision
                        # ------------------------------------

                        path_collision = (
                            check_path_collision(

                                current_position,

                                predicted_position,

                                collision_zone

                            )
                        )


                        # ------------------------------------
                        # Visual TTC
                        # ------------------------------------

                        visual_ttc = (
                            calculate_visual_ttc(
                                history
                            )
                        )


                        # ------------------------------------
                        # Risk
                        # ------------------------------------

                        risk_level = (
                            determine_risk_level(

                                approaching,

                                path_collision,

                                visual_ttc

                            )
                        )


                        # ------------------------------------
                        # Hazard 저장
                        # 현재 프레임의 객체만 포함
                        # ------------------------------------

                        hazards.append({

                            "track_id": track_id,

                            "class_name": class_name,

                            "approach_rate": approach_rate,

                            "approaching": approaching,

                            "path_collision": path_collision,

                            "visual_ttc": visual_ttc,

                            "risk_level": risk_level

                        })


                    # =================================================
                    # 13. 위험도 변화 디버깅
                    # =================================================

                    if video_name == DEBUG_VIDEO:

                        previous_risk = (
                            last_risk_by_id.get(
                                track_id,
                                "UNKNOWN"
                            )
                        )

                        # 위험도가 변경된 순간만 출력
                        if previous_risk != risk_level:

                            # 경고가 발생하거나 해제될 때만
                            if (

                                risk_level in (
                                    "CAUTION",
                                    "DANGER"
                                )

                                or

                                previous_risk in (
                                    "CAUTION",
                                    "DANGER"
                                )

                            ):

                                print(
                                    "\n-----------------------------------"
                                )

                                print(
                                    "[RISK CHANGE]"
                                )

                                print(
                                    "Frame:",
                                    frame_index
                                )

                                print(
                                    "Time:",
                                    round(timestamp, 3)
                                )

                                print(
                                    "Track ID:",
                                    track_id
                                )

                                print(
                                    "Class:",
                                    class_name
                                )

                                print(
                                    "Risk:",
                                    previous_risk,
                                    "->",
                                    risk_level
                                )

                                print(
                                    "ApproachRate:",
                                    round(
                                        approach_rate,
                                        4
                                    )
                                )

                                print(
                                    "LongApproachRate:",
                                    round(
                                        long_approach_rate,
                                        4
                                    )
                                )

                                print(
                                    "Approaching:",
                                    approaching
                                )

                                print(
                                    "PathCollision:",
                                    path_collision
                                )

                                print(
                                    "Visual TTC:",
                                    visual_ttc
                                )

                                print(
                                    "BBox Area:",
                                    round(
                                        current_area,
                                        2
                                    )
                                )

                                print(
                                    "History Length:",
                                    len(history)
                                )

                                print(
                                    "-----------------------------------"
                                )


                        # 현재 위험도를 기록
                        last_risk_by_id[
                            track_id
                        ] = risk_level


                    # =================================================
                    # 14. CSV 기록
                    # =================================================

                    if predicted_position is None:

                        predicted_x = ""
                        predicted_y = ""

                    else:

                        predicted_x = (
                            predicted_position[0]
                        )

                        predicted_y = (
                            predicted_position[1]
                        )


                    csv_writer.writerow([

                        frame_index,

                        timestamp,

                        track_id,

                        class_name,

                        current_area,

                        approach_rate,

                        long_approach_rate,

                        approaching,

                        path_collision,

                        visual_ttc,

                        risk_level,

                        center_x,

                        center_y,

                        predicted_x,

                        predicted_y

                    ])


                    # =================================================
                    # 15. 시각화 정보 저장
                    # =================================================

                    display_info.append({

                        "track_id": track_id,

                        "class_name": class_name,

                        "bbox": (
                            x1, y1, x2, y2
                        ),

                        "center_x": center_x,

                        "center_y": center_y,

                        "predicted_position": predicted_position,

                        "approach_rate": approach_rate,

                        "path_collision": path_collision,

                        "visual_ttc": visual_ttc,

                        "risk_level": risk_level

                    })


            # =================================================
            # 16. Primary Hazard
            # =================================================

            primary_hazard = (
                select_primary_hazard(
                    hazards
                )
            )


            # =================================================
            # 17. Collision Zone 시각화
            # =================================================

            zx1, zy1, zx2, zy2 = collision_zone

            cv2.rectangle(

                frame,

                (
                    int(zx1),
                    int(zy1)
                ),

                (
                    int(zx2),
                    int(zy2)
                ),

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


            # =================================================
            # 18. 객체별 시각화
            # =================================================

            for info in display_info:

                x1, y1, x2, y2 = (
                    info["bbox"]
                )

                track_id = (
                    info["track_id"]
                )

                risk_level = (
                    info["risk_level"]
                )

                color = RISK_COLORS[
                    risk_level
                ]


                # --------------------------------------------
                # 예측 이동경로 표시
                # --------------------------------------------

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


                    # 예상 도착점 표시
                    if (

                        0 <= end_point[0] < frame_width

                        and

                        0 <= end_point[1] < frame_height

                    ):

                        cv2.circle(

                            frame,

                            end_point,

                            6,

                            (255, 0, 0),

                            -1

                        )


                # --------------------------------------------
                # Primary Hazard 강조
                # --------------------------------------------

                thickness = 2

                if (

                    primary_hazard is not None

                    and

                    track_id == primary_hazard["track_id"]

                ):

                    thickness = 4


                # --------------------------------------------
                # Bounding Box
                # --------------------------------------------

                cv2.rectangle(

                    frame,

                    (
                        int(x1),
                        int(y1)
                    ),

                    (
                        int(x2),
                        int(y2)
                    ),

                    color,

                    thickness

                )


                # --------------------------------------------
                # 간단한 Label
                # --------------------------------------------

                label = (

                    f"ID:{track_id} "

                    f"{info['class_name']} "

                    f"{risk_level}"

                )


                text_x = max(
                    0,
                    int(x1)
                )

                text_y = max(
                    90,
                    int(y1) - 10
                )


                cv2.putText(

                    frame,

                    label,

                    (
                        text_x,
                        text_y
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    color,

                    2

                )


            # =================================================
            # 19. 상단 정보 표시
            # =================================================

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


            # -----------------------------------------------
            # Primary Hazard 표시
            # -----------------------------------------------

            if primary_hazard is None:

                primary_text = (
                    "Primary Hazard: NONE"
                )

                primary_color = (
                    0, 255, 0
                )

            else:

                primary_text = (

                    f"Primary Hazard: "

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


            # =================================================
            # 20. 결과 프레임 저장
            # =================================================

            writer.write(
                frame
            )


    # ========================================================
    # 21. 파일 종료
    # ========================================================

    finally:

        writer.release()

        csv_file.close()


    if frame_index == 0:

        raise RuntimeError(
            "처리된 프레임이 없습니다."
        )


    print("\n===================================")

    print(
        "영상 분석 완료:",
        video_name
    )

    print(
        "처리 프레임:",
        frame_index
    )

    print(
        "결과 영상:",
        output_path
    )

    print(
        "CSV 로그:",
        csv_path
    )

    print("===================================")


    return output_path


# ============================================================
# 22. 전체 영상 순차 실행
# ============================================================

success = []
failed = []


for video_name in VIDEO_NAMES:

    try:

        process_video(
            video_name
        )

        success.append(
            video_name
        )

    except Exception as e:

        print(
            "분석 실패:",
            video_name
        )

        print(
            "오류:",
            e
        )

        failed.append(
            video_name
        )


# ============================================================
# 23. 최종 결과
# ============================================================

print("\n===================================")

print("AI1 + AI2 통합 테스트 종료")

print("===================================")

print(
    "성공:",
    len(success)
)

print(
    "실패:",
    len(failed)
)


print("\n성공 영상:")

for name in success:

    print("-", name)


print("\n실패 영상:")

for name in failed:

    print("-", name)


print("===================================")


if failed:

    raise SystemExit(1)