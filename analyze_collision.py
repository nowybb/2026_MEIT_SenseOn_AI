from pathlib import Path
from collections import defaultdict
import csv
import math

from ultralytics import YOLO

from ai2.track_history import TrackHistory
from ai2.approach import is_approaching
from ai2.trajectory import predict_future_position
from ai2.collision import (
    create_collision_zone,
    check_path_collision,
    check_bbox_path_collision,
    calculate_bbox_zone_overlap_ratio,
    predict_future_bbox,
)


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI1_DIR = BASE_DIR / "ai1"
VIDEO_DIR = AI1_DIR / "videos"
OUTPUT_DIR = AI1_DIR / "outputs" / "collision_evaluation"

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

# 현재 실제 Risk pipeline과 동일
FUTURE_TIME = 1.0

# bbox overlap threshold 후보
OVERLAP_THRESHOLDS = [
    0.01,
    0.05,
    0.10,
    0.20,
]


# ============================================================
# 2. 영상 하나 분석
# ============================================================

def evaluate_video(video_name):

    video_path = VIDEO_DIR / video_name

    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}"
        )

    print("\n===================================")
    print("Collision 평가 시작:", video_name)
    print("===================================")

    # --------------------------------------------------------
    # 영상 정보
    # --------------------------------------------------------

    import cv2

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

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            "영상 크기를 확인할 수 없습니다."
        )

    print("FPS:", fps)

    print(
        "Resolution:",
        frame_width,
        "x",
        frame_height,
    )

    # --------------------------------------------------------
    # Collision Zone
    # --------------------------------------------------------

    collision_zone = create_collision_zone(
        frame_width,
        frame_height,
    )

    print(
        "Collision Zone:",
        collision_zone,
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
    # Track History
    # --------------------------------------------------------

    track_history = TrackHistory(
        max_history=HISTORY_SIZE
    )

    frame_index = 0

    rows = []

    # threshold별 통계
    statistics = {}

    for threshold in OVERLAP_THRESHOLDS:

        statistics[threshold] = {
            "total": 0,
            "both_false": 0,
            "both_true": 0,
            "center_only": 0,
            "bbox_only": 0,
        }

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

            # ------------------------------------------------
            # Detection 정보
            # ------------------------------------------------

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

            bbox = [
                x1,
                y1,
                x2,
                y2,
            ]

            detection = {
                "track_id": track_id,
                "class_name": class_name,
                "confidence": confidence,
                "timestamp": timestamp,
                "bbox": bbox,
                "center_x": center_x,
                "center_y": center_y,
            }

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            track_history.update(
                detection
            )

            history = (
                track_history.get_history(
                    track_id
                )
            )

            if (
                len(history) < 3
                or (
                    history[-1]["timestamp"]
                    - history[0]["timestamp"]
                ) < 0.1
            ):
                continue

            # ------------------------------------------------
            # Approach
            # ------------------------------------------------

            approaching = (
                is_approaching(
                    history
                )
            )

            # ------------------------------------------------
            # 동일한 Trajectory 사용
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
            # 기존 center collision
            # ------------------------------------------------

            center_collision = (
                check_path_collision(
                    current_position,
                    predicted_position,
                    collision_zone,
                )
            )

            # ------------------------------------------------
            # 미래 bbox 및 최종 overlap 참고값
            # ------------------------------------------------

            future_bbox = (
                predict_future_bbox(
                    bbox,
                    current_position,
                    predicted_position,
                )
            )

            if future_bbox is None:
                future_overlap_ratio = 0.0

            else:
                future_overlap_ratio = (
                    calculate_bbox_zone_overlap_ratio(
                        future_bbox,
                        collision_zone,
                    )
                )

            # ------------------------------------------------
            # threshold별 bbox collision
            # ------------------------------------------------

            bbox_results = {}

            for threshold in OVERLAP_THRESHOLDS:

                bbox_collision = (
                    check_bbox_path_collision(
                        current_bbox=bbox,
                        current_position=current_position,
                        predicted_position=predicted_position,
                        collision_zone=collision_zone,
                        overlap_threshold=threshold,
                    )
                )

                bbox_results[
                    threshold
                ] = bbox_collision

                stats = statistics[
                    threshold
                ]

                stats["total"] += 1

                if (
                    center_collision
                    and bbox_collision
                ):

                    stats[
                        "both_true"
                    ] += 1

                elif (
                    not center_collision
                    and not bbox_collision
                ):

                    stats[
                        "both_false"
                    ] += 1

                elif (
                    center_collision
                    and not bbox_collision
                ):

                    stats[
                        "center_only"
                    ] += 1

                else:

                    stats[
                        "bbox_only"
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

                "center_x": center_x,
                "center_y": center_y,

                "predicted_x":
                    predicted_position[0],

                "predicted_y":
                    predicted_position[1],

                "future_bbox_overlap":
                    future_overlap_ratio,

                "center_collision":
                    center_collision,

                "bbox_collision_001":
                    bbox_results[0.01],

                "bbox_collision_005":
                    bbox_results[0.05],

                "bbox_collision_010":
                    bbox_results[0.10],

                "bbox_collision_020":
                    bbox_results[0.20],
            })

    # ========================================================
    # 4. CSV 저장
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_collision_evaluation.csv"
    )

    fieldnames = [
        "frame",
        "timestamp",
        "track_id",
        "class_name",
        "approaching",
        "center_x",
        "center_y",
        "predicted_x",
        "predicted_y",
        "future_bbox_overlap",
        "center_collision",
        "bbox_collision_001",
        "bbox_collision_005",
        "bbox_collision_010",
        "bbox_collision_020",
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

    print(
        "\n평가 샘플 수:",
        len(rows)
    )

    for threshold in OVERLAP_THRESHOLDS:

        stats = statistics[
            threshold
        ]

        total = stats[
            "total"
        ]

        if total == 0:
            continue

        agreement = (
            stats["both_true"]
            + stats["both_false"]
        ) / total

        print(
            "\n-----------------------------------"
        )

        print(
            "BBox threshold:",
            threshold
        )

        print(
            "Agreement:",
            round(
                agreement * 100,
                2
            ),
            "%"
        )

        print(
            "Both False:",
            stats["both_false"]
        )

        print(
            "Both True:",
            stats["both_true"]
        )

        print(
            "Center Only:",
            stats["center_only"]
        )

        print(
            "BBox Only:",
            stats["bbox_only"]
        )

    print(
        "\nCSV:",
        csv_path
    )

    return statistics


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
print("전체 Collision 평가 결과")
print("===================================")

for video_name, results in all_results.items():

    print(
        f"\n{video_name}"
    )

    for threshold in OVERLAP_THRESHOLDS:

        stats = results[
            threshold
        ]

        total = stats[
            "total"
        ]

        if total == 0:
            continue

        agreement = (
            stats["both_true"]
            + stats["both_false"]
        ) / total

        print(
            f"  threshold={threshold:.2f}"
            f" | agreement={agreement * 100:.2f}%"
            f" | center_only={stats['center_only']}"
            f" | bbox_only={stats['bbox_only']}"
        )

print("\n===================================")