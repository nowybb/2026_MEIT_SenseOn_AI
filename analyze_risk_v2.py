from pathlib import Path
import csv
import math
import cv2

from ultralytics import YOLO

from ai2.track_history import TrackHistory
from ai2.approach import is_approaching
from ai2.trajectory import predict_future_position
from ai2.collision import (
    create_collision_zone,
    check_path_collision,
)
from ai2.ttc import calculate_visual_ttc_robust
from ai2.risk import (
    determine_risk_level,
    RiskStabilizer,
    RiskStabilizerV2,
)


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI1_DIR = BASE_DIR / "ai1"
VIDEO_DIR = AI1_DIR / "videos"
OUTPUT_DIR = AI1_DIR / "outputs" / "risk_v2_evaluation"

MODEL_PATH = "yolo11n.pt"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEO_NAMES = [
    "test1.mp4",
    "test2.mp4",
    "test3.mp4",
    "test4.mp4",
    "test5.mp4",
    "test_receding.mp4",
]

TARGET_CLASSES = [1, 2, 3]
CONFIDENCE = 0.3

HISTORY_SIZE = 10
FUTURE_TIME = 1.0

# V1과 V2 모두 동일하게 3프레임
RELEASE_FRAMES = 3


# ============================================================
# 2. 영상 하나 평가
# ============================================================

def evaluate_video(video_name):

    video_path = VIDEO_DIR / video_name

    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}"
        )

    print("\n===================================")
    print("Risk V1 vs V2 평가:", video_name)
    print("===================================")

    # --------------------------------------------------------
    # 영상 정보
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
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

    print("FPS:", fps)

    # --------------------------------------------------------
    # Collision Zone
    # --------------------------------------------------------

    collision_zone = create_collision_zone(
        frame_width,
        frame_height,
    )

    # --------------------------------------------------------
    # YOLO + BoT-SORT
    # --------------------------------------------------------

    model = YOLO(
        MODEL_PATH
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

    # --------------------------------------------------------
    # 상태 객체
    # --------------------------------------------------------

    track_history = TrackHistory(
        max_history=HISTORY_SIZE
    )

    stabilizer_v1 = RiskStabilizer(
        release_frames=RELEASE_FRAMES
    )

    stabilizer_v2 = RiskStabilizerV2(
        release_frames=RELEASE_FRAMES
    )

    previous_v1 = {}
    previous_v2 = {}

    stats = {
        "samples": 0,

        "v1_safe": 0,
        "v1_caution": 0,
        "v1_danger": 0,
        "v1_changes": 0,

        "v2_safe": 0,
        "v2_caution": 0,
        "v2_danger": 0,
        "v2_changes": 0,

        "different_frames": 0,
    }

    rows = []

    frame_index = 0

    # ========================================================
    # 3. 프레임 처리
    # ========================================================

    for result in results:

        frame_index += 1

        timestamp = (
            frame_index - 1
        ) / fps

        boxes = result.boxes

        if (
            boxes is None
            or boxes.id is None
        ):
            continue

        for box in boxes:

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

            detection = {
                "track_id": track_id,
                "class_name": class_name,
                "confidence": confidence,
                "timestamp": timestamp,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
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

            if len(history) < 3:
                continue

            duration = (
                history[-1]["timestamp"]
                - history[0]["timestamp"]
            )

            if duration < 0.1:
                continue

            # ------------------------------------------------
            # Approach
            # ------------------------------------------------

            approaching = is_approaching(
                history
            )

            # ------------------------------------------------
            # 기존 trajectory baseline 유지
            # ------------------------------------------------

            current_position = (
                center_x,
                center_y,
            )

            predicted_position = (
                predict_future_position(
                    history,
                    future_time=FUTURE_TIME,
                )
            )

            if predicted_position is None:
                continue

            # ------------------------------------------------
            # 기존 center collision 유지
            # ------------------------------------------------

            path_collision = (
                check_path_collision(
                    current_position,
                    predicted_position,
                    collision_zone,
                )
            )

            # ------------------------------------------------
            # Robust TTC
            # ------------------------------------------------

            visual_ttc = (
                calculate_visual_ttc_robust(
                    history
                )
            )

            # ------------------------------------------------
            # Raw Risk
            # ------------------------------------------------

            raw_risk = (
                determine_risk_level(
                    approaching,
                    path_collision,
                    visual_ttc,
                )
            )

            # ------------------------------------------------
            # V1 / V2
            # ------------------------------------------------

            risk_v1 = stabilizer_v1.update(
                track_id,
                raw_risk,
            )

            risk_v2 = stabilizer_v2.update(
                track_id,
                raw_risk,
            )

            stats["samples"] += 1

            stats[
                f"v1_{risk_v1.lower()}"
            ] += 1

            stats[
                f"v2_{risk_v2.lower()}"
            ] += 1

            # ------------------------------------------------
            # 상태 변화
            # ------------------------------------------------

            v1_changed = False
            v2_changed = False

            if track_id in previous_v1:

                if (
                    previous_v1[track_id]
                    != risk_v1
                ):
                    v1_changed = True
                    stats["v1_changes"] += 1

            if track_id in previous_v2:

                if (
                    previous_v2[track_id]
                    != risk_v2
                ):
                    v2_changed = True
                    stats["v2_changes"] += 1

            previous_v1[
                track_id
            ] = risk_v1

            previous_v2[
                track_id
            ] = risk_v2

            # ------------------------------------------------
            # V1 / V2 불일치
            # ------------------------------------------------

            different = (
                risk_v1 != risk_v2
            )

            if different:
                stats[
                    "different_frames"
                ] += 1

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            rows.append({
                "frame": frame_index,
                "timestamp": timestamp,
                "track_id": track_id,
                "class_name": class_name,

                "approaching": approaching,
                "path_collision": path_collision,
                "visual_ttc": visual_ttc,

                "raw_risk": raw_risk,

                "risk_v1": risk_v1,
                "risk_v2": risk_v2,

                "v1_changed": v1_changed,
                "v2_changed": v2_changed,

                "different": different,
            })

    # ========================================================
    # 4. CSV 저장
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_risk_v1_vs_v2.csv"
    )

    fieldnames = [
        "frame",
        "timestamp",
        "track_id",
        "class_name",
        "approaching",
        "path_collision",
        "visual_ttc",
        "raw_risk",
        "risk_v1",
        "risk_v2",
        "v1_changed",
        "v2_changed",
        "different",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # 5. 결과 출력
    # ========================================================

    v1_danger_seconds = (
        stats["v1_danger"]
        / fps
    )

    v2_danger_seconds = (
        stats["v2_danger"]
        / fps
    )

    difference_ratio = (
        stats["different_frames"]
        / stats["samples"] * 100
        if stats["samples"] > 0
        else 0.0
    )

    print(
        "\nSamples:",
        stats["samples"]
    )

    print("\n[V1]")

    print(
        "SAFE:",
        stats["v1_safe"]
    )

    print(
        "CAUTION:",
        stats["v1_caution"]
    )

    print(
        "DANGER:",
        stats["v1_danger"]
    )

    print(
        "State Changes:",
        stats["v1_changes"]
    )

    print(
        "DANGER Seconds:",
        round(v1_danger_seconds, 2)
    )

    print("\n[V2]")

    print(
        "SAFE:",
        stats["v2_safe"]
    )

    print(
        "CAUTION:",
        stats["v2_caution"]
    )

    print(
        "DANGER:",
        stats["v2_danger"]
    )

    print(
        "State Changes:",
        stats["v2_changes"]
    )

    print(
        "DANGER Seconds:",
        round(v2_danger_seconds, 2)
    )

    print(
        "\nDifferent Frames:",
        stats["different_frames"]
    )

    print(
        "Difference Ratio:",
        round(difference_ratio, 2),
        "%"
    )

    print(
        "\nCSV:",
        csv_path
    )

    return {
        "fps": fps,
        "stats": stats,
    }


# ============================================================
# 6. 전체 영상 실행
# ============================================================

all_results = {}

for video_name in VIDEO_NAMES:

    try:

        result = evaluate_video(
            video_name
        )

        all_results[
            video_name
        ] = result

    except Exception as e:

        print(
            "\n평가 실패:",
            video_name
        )

        print(
            "오류:",
            e
        )


# ============================================================
# 7. 전체 결과
# ============================================================

print("\n\n===================================")
print("전체 Risk V1 vs V2 평가 결과")
print("===================================")

total_samples = 0
total_v1_changes = 0
total_v2_changes = 0
total_v1_danger = 0
total_v2_danger = 0
total_different = 0

for video_name, result in all_results.items():

    stats = result["stats"]
    fps = result["fps"]

    total_samples += stats["samples"]

    total_v1_changes += (
        stats["v1_changes"]
    )

    total_v2_changes += (
        stats["v2_changes"]
    )

    total_v1_danger += (
        stats["v1_danger"]
    )

    total_v2_danger += (
        stats["v2_danger"]
    )

    total_different += (
        stats["different_frames"]
    )

    print(
        f"\n{video_name}"
    )

    print(
        f"  V1 | changes={stats['v1_changes']:4d}"
        f" | SAFE={stats['v1_safe']:4d}"
        f" | CAUTION={stats['v1_caution']:4d}"
        f" | DANGER={stats['v1_danger']:4d}"
        f" | danger_sec={stats['v1_danger'] / fps:.2f}"
    )

    print(
        f"  V2 | changes={stats['v2_changes']:4d}"
        f" | SAFE={stats['v2_safe']:4d}"
        f" | CAUTION={stats['v2_caution']:4d}"
        f" | DANGER={stats['v2_danger']:4d}"
        f" | danger_sec={stats['v2_danger'] / fps:.2f}"
    )

    print(
        f"  Different Frames:"
        f" {stats['different_frames']}"
    )


print("\n-----------------------------------")
print("TOTAL")
print("-----------------------------------")

print(
    "Samples:",
    total_samples
)

print(
    "V1 State Changes:",
    total_v1_changes
)

print(
    "V2 State Changes:",
    total_v2_changes
)

print(
    "V1 DANGER Frames:",
    total_v1_danger
)

print(
    "V2 DANGER Frames:",
    total_v2_danger
)

print(
    "Different Frames:",
    total_different
)

if total_samples > 0:

    print(
        "Difference Ratio:",
        round(
            total_different
            / total_samples
            * 100,
            2
        ),
        "%"
    )

print("\n===================================")