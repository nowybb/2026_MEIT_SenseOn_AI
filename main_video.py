# main_video.py
# 기존 위험도 판단은 유지하고, 기존/개선 후보 궤적을 영상과 CSV에서 비교한다.
# BLE 송신은 하지 않는다. 비교 결과는 outputs/trajectory_comparison에 저장한다.

from pathlib import Path
import csv
import math

import cv2
from ultralytics import YOLO

from ai2.track_history import TrackHistory
from ai2.approach import calculate_bbox_area, calculate_approach_rate, is_approaching
from ai2.trajectory import predict_future_position, predict_stable_position
from ai2.collision import (
    create_collision_zone,
    check_path_collision,
    calculate_bbox_zone_overlap,
    check_bbox_path_collision,
)
from ai2.proximity import calculate_proximity_features
from ai2.ttc import calculate_visual_ttc
from ai2.risk import determine_risk_level, select_primary_hazard


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI1_DIR = BASE_DIR / "ai1"
VIDEO_DIR = AI1_DIR / "videos"
OUTPUT_DIR = AI1_DIR / "outputs"
MODEL_PATH = AI1_DIR / "yolo11n.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARE_DIR = OUTPUT_DIR / "trajectory_comparison"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

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
STABLE_FUTURE_TIME = 0.3

CONFIDENCE = 0.3

# bicycle, car, motorcycle, bus, truck
TARGET_CLASSES = [1, 2, 3, 5, 7]

DEBUG_VIDEO = "test_receding.mp4"

# OpenCV BGR
RISK_COLORS = {
    "SAFE": (0, 255, 0),
    "CAUTION": (0, 255, 255),
    "DANGER": (0, 0, 255),
    "UNKNOWN": (200, 200, 200),
}


# 기존 CSV 20개 열 유지 + 비교용 3개 열 추가
CSV_COLUMNS = [
    "frame",
    "timestamp",
    "track_id",
    "class_name",
    "bbox_area",

    "approach_rate",
    "long_approach_rate",
    "approaching",

    "path_collision",
    "bbox_path_collision",
    "bbox_zone_overlap",

    "height_ratio",
    "area_ratio",
    "bottom_y_ratio",

    "visual_ttc",
    "risk_level",

    "center_x",
    "center_y",

    "predicted_x",
    "predicted_y",

    # 개선 후보 분석용
    "stable_predicted_x",
    "stable_predicted_y",
    "stable_path_collision",
]


# ============================================================
# 장기 접근률: 기존 디버깅 지표
# ============================================================

def calculate_long_approach_rate(history):

    if len(history) < 2:
        return 0.0

    first_area = calculate_bbox_area(
        history[0]["bbox"]
    )

    last_area = calculate_bbox_area(
        history[-1]["bbox"]
    )

    dt = (
        history[-1]["timestamp"]
        - history[0]["timestamp"]
    )

    if first_area <= 0 or dt <= 0:
        return 0.0

    return (
        (last_area - first_area) / first_area
    ) / dt


# ============================================================
# 방향 판단: 화면상 위치 기준
# ============================================================

def get_direction(center_x, frame_width):

    if center_x < frame_width / 3:
        return "LEFT"

    if center_x < frame_width * 2 / 3:
        return "CENTER"

    return "RIGHT"


# ============================================================
# 화살표 시각화
# ============================================================

def draw_arrow(
    frame,
    current_position,
    predicted_position,
    color,
    frame_width,
    frame_height,
):

    if predicted_position is None:
        return

    if not all(
        math.isfinite(float(v))
        for v in predicted_position
    ):
        return

    start = tuple(
        int(v) for v in current_position
    )

    end = tuple(
        int(v) for v in predicted_position
    )

    cv2.arrowedLine(
        frame,
        start,
        end,
        color,
        2,
        tipLength=0.15,
    )

    if (
        0 <= end[0] < frame_width
        and 0 <= end[1] < frame_height
    ):
        cv2.circle(
            frame,
            end,
            5,
            color,
            -1,
        )


# ============================================================
# 영상 하나 분석
# ============================================================

def process_video(video_name):

    video_path = VIDEO_DIR / video_name

    output_path = (
        COMPARE_DIR
        / f"{video_path.stem}_risk_result.mp4"
    )

    csv_path = (
        COMPARE_DIR
        / f"{video_path.stem}_risk_log.csv"
    )

    # --------------------------------------------------------
    # 파일 확인
    # --------------------------------------------------------

    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # 영상 정보 확인
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        cap.release()

        raise RuntimeError(
            f"영상을 열 수 없습니다: {video_path}"
        )

    try:
        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

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

    # --------------------------------------------------------
    # 결과 영상 저장
    # --------------------------------------------------------

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        writer.release()

        raise RuntimeError(
            f"결과 영상을 생성할 수 없습니다: {output_path}"
        )

    frame_index = 0

    print(
        f"\n분석 시작: {video_name} / "
        f"{fps:.2f} FPS / "
        f"{frame_width}x{frame_height}"
    )

    # ========================================================
    # 실제 영상 분석
    # ========================================================

    try:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            csv_writer = csv.writer(
                csv_file
            )

            csv_writer.writerow(
                CSV_COLUMNS
            )

            # ------------------------------------------------
            # YOLO / Tracking 초기화
            # ------------------------------------------------

            model = YOLO(
                str(MODEL_PATH)
            )

            track_history = TrackHistory(
                max_history=HISTORY_SIZE
            )

            last_risk_by_id = {}

            collision_zone = create_collision_zone(
                frame_width,
                frame_height,
            )

            print(
                "Collision Zone:",
                collision_zone,
            )

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
            # 프레임별 처리
            # =================================================

            for result in results:

                frame_index += 1

                frame = result.orig_img.copy()

                timestamp = (
                    frame_index - 1
                ) / fps

                hazards = []
                display_info = []

                boxes = result.boxes

                # =============================================
                # 객체별 분석
                # =============================================

                if (
                    boxes is not None
                    and boxes.id is not None
                ):

                    for box in boxes:

                        # -------------------------------------
                        # AI1 객체 탐지 정보
                        # -------------------------------------

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
                            box.xyxy[0],
                        )

                        bbox = [
                            x1,
                            y1,
                            x2,
                            y2,
                        ]

                        center_x = (
                            x1 + x2
                        ) / 2

                        center_y = (
                            y1 + y2
                        ) / 2

                        current_position = (
                            center_x,
                            center_y,
                        )

                        # -------------------------------------
                        # AI1 -> AI2 내부 데이터
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

                        track_history.update(
                            detection
                        )

                        history = (
                            track_history.get_history(
                                track_id
                            )
                        )

                        # -------------------------------------
                        # bbox 면적 및 근접도
                        # -------------------------------------

                        current_area = (
                            calculate_bbox_area(
                                bbox
                            )
                        )

                        proximity = (
                            calculate_proximity_features(
                                bbox=bbox,
                                frame_width=frame_width,
                                frame_height=frame_height,
                            )
                        )

                        bbox_zone_overlap = (
                            calculate_bbox_zone_overlap(
                                bbox,
                                collision_zone,
                            )
                        )

                        # -------------------------------------
                        # AI2 초기값
                        # -------------------------------------

                        approach_rate = 0.0
                        long_approach_rate = 0.0

                        approaching = False

                        path_collision = False
                        bbox_path_collision = None

                        visual_ttc = None

                        predicted_position = None

                        stable_predicted_position = None
                        stable_path_collision = None

                        risk_level = "UNKNOWN"

                        # -------------------------------------
                        # 분석 가능한 기록인지 확인
                        # -------------------------------------

                        enough_history = (
                            len(history) >= 3
                            and (
                                history[-1]["timestamp"]
                                - history[0]["timestamp"]
                            ) >= 0.1
                        )

                        if enough_history:

                            # =================================
                            # 1. 접근률
                            # =================================

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

                            long_approach_rate = (
                                calculate_long_approach_rate(
                                    history
                                )
                            )

                            # =================================
                            # 2. 기존 Trajectory
                            #
                            # 현재 위험도 판단에 사용
                            # =================================

                            predicted_position = (
                                predict_future_position(
                                    history,
                                    future_time=FUTURE_TIME,
                                )
                            )

                            path_collision = (
                                check_path_collision(
                                    current_position,
                                    predicted_position,
                                    collision_zone,
                                )
                            )

                            bbox_path_collision = (
                                check_bbox_path_collision(
                                    bbox=bbox,
                                    current_position=current_position,
                                    predicted_position=predicted_position,
                                    collision_zone=collision_zone,
                                )
                            )

                            # =================================
                            # 3. 개선 후보 Trajectory
                            #
                            # 위험도 판단에는 사용하지 않음
                            # =================================

                            stable_predicted_position = (
                                predict_stable_position(
                                    history,
                                    future_time=STABLE_FUTURE_TIME,
                                    window_size=HISTORY_SIZE,
                                )
                            )

                            if stable_predicted_position is not None:

                                stable_path_collision = (
                                    check_path_collision(
                                        current_position,
                                        stable_predicted_position,
                                        collision_zone,
                                    )
                                )

                            # =================================
                            # 4. Visual TTC
                            # =================================

                            visual_ttc = (
                                calculate_visual_ttc(
                                    history
                                )
                            )

                            # =================================
                            # 5. 실제 위험도 판단
                            #
                            # 기존 path_collision 사용
                            # stable_path_collision 사용 X
                            # =================================

                            risk_level = (
                                determine_risk_level(
                                    approaching,
                                    path_collision,
                                    visual_ttc,
                                )
                            )

                            # =================================
                            # 6. 현재 객체 저장
                            # =================================

                            hazards.append({
                                "track_id": track_id,
                                "class_name": class_name,
                                "center_x": center_x,
                                "approach_rate": approach_rate,
                                "approaching": approaching,
                                "path_collision": path_collision,
                                "visual_ttc": visual_ttc,
                                "risk_level": risk_level,
                            })

                        # =====================================
                        # 위험도 변화 디버깅
                        # =====================================

                        if video_name == DEBUG_VIDEO:

                            previous_risk = (
                                last_risk_by_id.get(
                                    track_id,
                                    "UNKNOWN",
                                )
                            )

                            if (
                                previous_risk != risk_level
                                and (
                                    previous_risk in (
                                        "CAUTION",
                                        "DANGER",
                                    )
                                    or risk_level in (
                                        "CAUTION",
                                        "DANGER",
                                    )
                                )
                            ):

                                print(
                                    f"[RISK CHANGE] "
                                    f"frame={frame_index} "
                                    f"time={timestamp:.3f}s "
                                    f"ID={track_id} "
                                    f"{previous_risk}->{risk_level} "
                                    f"approach={approach_rate:.4f} "
                                    f"path={path_collision} "
                                    f"stable_path={stable_path_collision} "
                                    f"TTC={visual_ttc}"
                                )

                            last_risk_by_id[
                                track_id
                            ] = risk_level

                        # =====================================
                        # CSV 기록
                        # =====================================

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

                            (
                                bbox_path_collision
                                if bbox_path_collision is not None
                                else ""
                            ),

                            bbox_zone_overlap,

                            proximity["height_ratio"],
                            proximity["area_ratio"],
                            proximity["bottom_y_ratio"],

                            (
                                visual_ttc
                                if visual_ttc is not None
                                else ""
                            ),

                            risk_level,

                            center_x,
                            center_y,

                            (
                                predicted_position[0]
                                if predicted_position is not None
                                else ""
                            ),

                            (
                                predicted_position[1]
                                if predicted_position is not None
                                else ""
                            ),

                            (
                                stable_predicted_position[0]
                                if stable_predicted_position is not None
                                else ""
                            ),

                            (
                                stable_predicted_position[1]
                                if stable_predicted_position is not None
                                else ""
                            ),

                            (
                                stable_path_collision
                                if stable_path_collision is not None
                                else ""
                            ),
                        ])

                        # =====================================
                        # 시각화 정보
                        # =====================================

                        display_info.append({
                            "track_id": track_id,
                            "class_name": class_name,
                            "bbox": bbox,
                            "center": current_position,
                            "predicted": predicted_position,
                            "stable_predicted": stable_predicted_position,
                            "risk": risk_level,
                        })

                # =============================================
                # Primary Hazard
                # =============================================

                primary_hazard = (
                    select_primary_hazard(
                        hazards
                    )
                )

                # =============================================
                # AI 최종 출력 규격
                #
                # BLE 전송은 아직 하지 않음
                # =============================================

                if primary_hazard is None:

                    final_result = None

                else:

                    final_result = {
                        "object": primary_hazard["class_name"],
                        "direction": get_direction(
                            primary_hazard["center_x"],
                            frame_width,
                        ),
                        "risk": primary_hazard["risk_level"],
                        "ttc": primary_hazard["visual_ttc"],
                    }

                # =============================================
                # Collision Zone 표시
                # =============================================

                zx1, zy1, zx2, zy2 = collision_zone

                cv2.rectangle(
                    frame,
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

                cv2.putText(
                    frame,
                    "Collision Zone (Proxy)",
                    (
                        int(zx1),
                        min(
                            frame_height - 10,
                            int(zy1) + 20,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

                # =============================================
                # 객체별 시각화
                # =============================================

                for info in display_info:

                    x1, y1, x2, y2 = info["bbox"]

                    color = RISK_COLORS[
                        info["risk"]
                    ]

                    # -----------------------------------------
                    # 파란색: 기존 1초 예측
                    # -----------------------------------------

                    draw_arrow(
                        frame,
                        info["center"],
                        info["predicted"],
                        (255, 0, 0),
                        frame_width,
                        frame_height,
                    )

                    # -----------------------------------------
                    # 자주색: 개선 후보 0.3초 예측
                    # -----------------------------------------

                    draw_arrow(
                        frame,
                        info["center"],
                        info["stable_predicted"],
                        (255, 0, 255),
                        frame_width,
                        frame_height,
                    )

                    # -----------------------------------------
                    # 주요 위험 객체 강조
                    # -----------------------------------------

                    is_primary = (
                        primary_hazard is not None
                        and info["track_id"]
                        == primary_hazard["track_id"]
                    )

                    cv2.rectangle(
                        frame,
                        (
                            int(x1),
                            int(y1),
                        ),
                        (
                            int(x2),
                            int(y2),
                        ),
                        color,
                        4 if is_primary else 2,
                    )

                    # -----------------------------------------
                    # 객체 정보 표시
                    # -----------------------------------------

                    label = (
                        f"ID:{info['track_id']} "
                        f"{info['class_name']} "
                        f"{info['risk']}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            max(0, int(x1)),
                            max(90, int(y1) - 10),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                # =============================================
                # 상단 정보 표시
                # =============================================

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame_width, 88),
                    (30, 30, 30),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Frame: {frame_index}",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if primary_hazard is None:

                    primary_text = (
                        "Primary Hazard: NONE"
                    )

                    primary_color = (
                        0,
                        255,
                        0,
                    )

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
                    2,
                )

                cv2.putText(
                    frame,
                    "BLUE: old 1.0s | MAGENTA: candidate 0.3s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

                # =============================================
                # 결과 영상 저장
                # =============================================

                writer.write(
                    frame
                )

    finally:

        writer.release()

    # ========================================================
    # 영상 분석 완료
    # ========================================================

    if frame_index == 0:
        raise RuntimeError(
            f"처리된 프레임이 없습니다: {video_name}"
        )

    print(
        f"분석 완료: {video_name} / "
        f"프레임 {frame_index}"
    )

    print(
        "결과 영상:",
        output_path,
    )

    print(
        "CSV 로그:",
        csv_path,
    )

    return output_path


# ============================================================
# 영상 6개 순차 실행
# ============================================================

def main():

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

        except Exception as exc:

            print(
                f"분석 실패: {video_name} / "
                f"{type(exc).__name__}: {exc}"
            )

            failed.append(
                video_name
            )

    print(
        "\nAI1 + AI2 통합 테스트 종료"
    )

    print(
        "성공:",
        len(success),
    )

    print(
        "실패:",
        len(failed),
    )

    if failed:

        print(
            "실패 영상:",
            ", ".join(failed),
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()