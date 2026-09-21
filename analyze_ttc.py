from pathlib import Path
from collections import defaultdict
import csv
import math

from ultralytics import YOLO

from ai2.track_history import TrackHistory
from ai2.ttc import (
    calculate_visual_ttc,
    calculate_visual_ttc_robust,
)


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI1_DIR = BASE_DIR / "ai1"
VIDEO_DIR = AI1_DIR / "videos"
OUTPUT_DIR = AI1_DIR / "outputs" / "ttc_evaluation"

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

DANGER_TTC = 2.0
CAUTION_TTC = 4.0


# ============================================================
# 2. TTC 상태 분류
# ============================================================

def classify_ttc(ttc):

    if ttc is None:
        return "NONE"

    if ttc <= DANGER_TTC:
        return "DANGER"

    if ttc <= CAUTION_TTC:
        return "CAUTION"

    return "SAFE"


# ============================================================
# 3. 영상 하나 분석
# ============================================================

def evaluate_video(video_name):

    video_path = VIDEO_DIR / video_name

    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}"
        )

    print("\n===================================")
    print("TTC 평가 시작:", video_name)
    print("===================================")

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    import cv2

    cap = cv2.VideoCapture(
        str(video_path)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            "영상 FPS를 확인할 수 없습니다."
        )

    print("FPS:", fps)

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

    track_history = TrackHistory(
        max_history=HISTORY_SIZE
    )

    frame_index = 0
    rows = []

    # track별 이전 상태
    previous_states = {}

    stats = {
        "samples": 0,

        "old_none": 0,
        "robust_none": 0,

        "old_danger": 0,
        "robust_danger": 0,

        "old_caution": 0,
        "robust_caution": 0,

        "old_safe": 0,
        "robust_safe": 0,

        "old_state_changes": 0,
        "robust_state_changes": 0,

        "old_threshold_crossings": 0,
        "robust_threshold_crossings": 0,

        "old_ttc_sum": 0.0,
        "robust_ttc_sum": 0.0,

        "old_ttc_count": 0,
        "robust_ttc_count": 0,
    }

    # ========================================================
    # 4. 프레임 처리
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

            # Robust 방식 최소 조건에 맞춤
            if len(history) < 3:
                continue

            duration = (
                history[-1]["timestamp"]
                - history[0]["timestamp"]
            )

            if duration < 0.1:
                continue

            # ------------------------------------------------
            # TTC 계산
            # ------------------------------------------------

            old_ttc = (
                calculate_visual_ttc(
                    history,
                    window_size=5,
                )
            )

            robust_ttc = (
                calculate_visual_ttc_robust(
                    history,
                    window_size=10,
                )
            )

            old_state = (
                classify_ttc(
                    old_ttc
                )
            )

            robust_state = (
                classify_ttc(
                    robust_ttc
                )
            )

            stats["samples"] += 1

            # ------------------------------------------------
            # 상태별 개수
            # ------------------------------------------------

            stats[
                f"old_{old_state.lower()}"
            ] += 1

            stats[
                f"robust_{robust_state.lower()}"
            ] += 1

            # ------------------------------------------------
            # TTC 평균용
            # ------------------------------------------------

            if old_ttc is not None:

                stats[
                    "old_ttc_sum"
                ] += old_ttc

                stats[
                    "old_ttc_count"
                ] += 1

            if robust_ttc is not None:

                stats[
                    "robust_ttc_sum"
                ] += robust_ttc

                stats[
                    "robust_ttc_count"
                ] += 1

            # ------------------------------------------------
            # 이전 상태와 비교
            # ------------------------------------------------

            previous = previous_states.get(
                track_id
            )

            old_changed = False
            robust_changed = False

            old_threshold_crossed = False
            robust_threshold_crossed = False

            if previous is not None:

                previous_old = previous[
                    "old"
                ]

                previous_robust = previous[
                    "robust"
                ]

                if old_state != previous_old:

                    old_changed = True

                    stats[
                        "old_state_changes"
                    ] += 1

                if robust_state != previous_robust:

                    robust_changed = True

                    stats[
                        "robust_state_changes"
                    ] += 1

                # NONE 전환은 제외하고
                # 실제 SAFE/CAUTION/DANGER 경계 이동만 계산
                if (
                    old_state != previous_old
                    and old_state != "NONE"
                    and previous_old != "NONE"
                ):

                    old_threshold_crossed = True

                    stats[
                        "old_threshold_crossings"
                    ] += 1

                if (
                    robust_state != previous_robust
                    and robust_state != "NONE"
                    and previous_robust != "NONE"
                ):

                    robust_threshold_crossed = True

                    stats[
                        "robust_threshold_crossings"
                    ] += 1

            previous_states[
                track_id
            ] = {
                "old": old_state,
                "robust": robust_state,
            }

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            rows.append({
                "frame": frame_index,
                "timestamp": timestamp,
                "track_id": track_id,
                "class_name": class_name,

                "old_ttc":
                    old_ttc,

                "robust_ttc":
                    robust_ttc,

                "old_state":
                    old_state,

                "robust_state":
                    robust_state,

                "old_state_changed":
                    old_changed,

                "robust_state_changed":
                    robust_changed,

                "old_threshold_crossed":
                    old_threshold_crossed,

                "robust_threshold_crossed":
                    robust_threshold_crossed,
            })

    # ========================================================
    # 5. CSV 저장
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_ttc_evaluation.csv"
    )

    fieldnames = [
        "frame",
        "timestamp",
        "track_id",
        "class_name",
        "old_ttc",
        "robust_ttc",
        "old_state",
        "robust_state",
        "old_state_changed",
        "robust_state_changed",
        "old_threshold_crossed",
        "robust_threshold_crossed",
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
    # 6. 평균 계산
    # ========================================================

    if stats["old_ttc_count"] > 0:

        old_mean = (
            stats["old_ttc_sum"]
            / stats["old_ttc_count"]
        )

    else:
        old_mean = None

    if stats["robust_ttc_count"] > 0:

        robust_mean = (
            stats["robust_ttc_sum"]
            / stats["robust_ttc_count"]
        )

    else:
        robust_mean = None

    stats["old_mean_ttc"] = old_mean
    stats["robust_mean_ttc"] = robust_mean

    # ========================================================
    # 7. 출력
    # ========================================================

    print(
        "\n평가 샘플 수:",
        stats["samples"]
    )

    print("\n[Old TTC]")

    print(
        "NONE:",
        stats["old_none"]
    )

    print(
        "SAFE:",
        stats["old_safe"]
    )

    print(
        "CAUTION:",
        stats["old_caution"]
    )

    print(
        "DANGER:",
        stats["old_danger"]
    )

    print(
        "State Changes:",
        stats["old_state_changes"]
    )

    print(
        "Threshold Crossings:",
        stats["old_threshold_crossings"]
    )

    print(
        "Mean TTC:",
        (
            round(old_mean, 2)
            if old_mean is not None
            else None
        )
    )

    print("\n[Robust TTC]")

    print(
        "NONE:",
        stats["robust_none"]
    )

    print(
        "SAFE:",
        stats["robust_safe"]
    )

    print(
        "CAUTION:",
        stats["robust_caution"]
    )

    print(
        "DANGER:",
        stats["robust_danger"]
    )

    print(
        "State Changes:",
        stats["robust_state_changes"]
    )

    print(
        "Threshold Crossings:",
        stats["robust_threshold_crossings"]
    )

    print(
        "Mean TTC:",
        (
            round(robust_mean, 2)
            if robust_mean is not None
            else None
        )
    )

    print(
        "\nCSV:",
        csv_path
    )

    return stats


# ============================================================
# 8. 전체 영상 실행
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
# 9. 전체 결과
# ============================================================

print("\n\n===================================")
print("전체 TTC 평가 결과")
print("===================================")

for video_name, stats in all_results.items():

    print(
        f"\n{video_name}"
    )

    print(
        f"  Samples            : {stats['samples']}"
    )

    print(
        f"  Old NONE           : {stats['old_none']}"
    )

    print(
        f"  Robust NONE        : {stats['robust_none']}"
    )

    print(
        f"  Old DANGER         : {stats['old_danger']}"
    )

    print(
        f"  Robust DANGER      : {stats['robust_danger']}"
    )

    print(
        f"  Old Changes        : {stats['old_state_changes']}"
    )

    print(
        f"  Robust Changes     : {stats['robust_state_changes']}"
    )

    print(
        f"  Old Crossings      : {stats['old_threshold_crossings']}"
    )

    print(
        f"  Robust Crossings   : {stats['robust_threshold_crossings']}"
    )

    print(
        f"  Old Mean TTC       : {stats['old_mean_ttc']}"
    )

    print(
        f"  Robust Mean TTC    : {stats['robust_mean_ttc']}"
    )

print("\n===================================")