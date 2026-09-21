from pathlib import Path
from collections import defaultdict
from statistics import mean, median
import csv
import math

from ultralytics import YOLO

from ai2.trajectory import (
    predict_future_position,
    predict_future_position_robust,
)


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI1_DIR = BASE_DIR / "ai1"
VIDEO_DIR = AI1_DIR / "videos"
OUTPUT_DIR = AI1_DIR / "outputs" / "trajectory_evaluation"

MODEL_PATH = AI1_DIR / "yolo11n.pt"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

# 두 알고리즘을 동일 조건에서 비교
FUTURE_TIME = 0.3


# ============================================================
# 2. 두 점 사이 거리
# ============================================================

def calculate_error(predicted, actual):
    """
    예측 위치와 실제 위치 사이의
    Euclidean distance를 계산한다.

    단위:
        pixel
    """

    px, py = predicted
    ax, ay = actual

    return math.sqrt(
        (px - ax) ** 2
        + (py - ay) ** 2
    )


# ============================================================
# 3. 영상 하나 평가
# ============================================================

def evaluate_video(video_name):

    video_path = VIDEO_DIR / video_name

    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상 파일이 없습니다: {video_path}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {MODEL_PATH}"
        )

    print("\n===================================")
    print("Trajectory 평가 시작:", video_name)
    print("===================================")

    # --------------------------------------------------------
    # 영상 FPS 확인
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

    # 0.3초와 가장 가까운 프레임 수
    future_frames = max(
        1,
        round(FUTURE_TIME * fps)
    )

    actual_future_time = (
        future_frames / fps
    )

    print("FPS:", fps)
    print(
        "평가 미래 시간:",
        round(actual_future_time, 4),
        "sec"
    )
    print(
        "미래 프레임 간격:",
        future_frames
    )

    # --------------------------------------------------------
    # YOLO + BoT-SORT
    # --------------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
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
    # track별 전체 탐지 기록 저장
    # --------------------------------------------------------

    tracks = defaultdict(list)

    frame_index = 0

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

            tracks[track_id].append({
                "frame": frame_index,
                "timestamp": timestamp,
                "track_id": track_id,
                "class_name": class_name,
                "center_x": center_x,
                "center_y": center_y,
            })

    # --------------------------------------------------------
    # 예측 정확도 평가
    # --------------------------------------------------------

    rows = []

    old_errors = []
    robust_errors = []

    for track_id, history in tracks.items():

        if len(history) < 3:
            continue

        # 현재 기록 위치
        for current_index in range(
            2,
            len(history)
        ):

            current = history[
                current_index
            ]

            target_frame = (
                current["frame"]
                + future_frames
            )

            # -----------------------------------------------
            # 같은 track_id에서 목표 프레임 탐색
            # -----------------------------------------------

            future_item = None

            for candidate in history[
                current_index + 1:
            ]:

                if (
                    candidate["frame"]
                    == target_frame
                ):
                    future_item = candidate
                    break

                if (
                    candidate["frame"]
                    > target_frame
                ):
                    break

            # 정확히 해당 미래 프레임에서
            # 객체가 추적되지 않았다면 평가 제외
            if future_item is None:
                continue

            # -----------------------------------------------
            # 현재 시점까지의 history
            # -----------------------------------------------

            past_history = history[
                :current_index + 1
            ]

            # -----------------------------------------------
            # 기존 방식
            # 동일하게 0.3초 기준으로 비교
            # -----------------------------------------------

            old_prediction = (
                predict_future_position(
                    past_history,
                    future_time=actual_future_time,
                    window_size=5,
                )
            )

            # -----------------------------------------------
            # Robust 방식
            # -----------------------------------------------

            robust_prediction = (
                predict_future_position_robust(
                    past_history,
                    future_time=actual_future_time,
                    window_size=10,
                )
            )

            if (
                old_prediction is None
                or robust_prediction is None
            ):
                continue

            actual_position = (
                future_item["center_x"],
                future_item["center_y"],
            )

            # -----------------------------------------------
            # Pixel Error
            # -----------------------------------------------

            old_error = calculate_error(
                old_prediction,
                actual_position,
            )

            robust_error = calculate_error(
                robust_prediction,
                actual_position,
            )

            old_errors.append(
                old_error
            )

            robust_errors.append(
                robust_error
            )

            rows.append({

                "video": video_name,

                "track_id": track_id,

                "class_name":
                    current["class_name"],

                "prediction_frame":
                    current["frame"],

                "target_frame":
                    future_item["frame"],

                "prediction_time":
                    current["timestamp"],

                "actual_time":
                    future_item["timestamp"],

                "old_pred_x":
                    old_prediction[0],

                "old_pred_y":
                    old_prediction[1],

                "robust_pred_x":
                    robust_prediction[0],

                "robust_pred_y":
                    robust_prediction[1],

                "actual_x":
                    actual_position[0],

                "actual_y":
                    actual_position[1],

                "old_error_px":
                    old_error,

                "robust_error_px":
                    robust_error,

            })

    # --------------------------------------------------------
    # CSV 저장
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_trajectory_evaluation.csv"
    )

    fieldnames = [
        "video",
        "track_id",
        "class_name",
        "prediction_frame",
        "target_frame",
        "prediction_time",
        "actual_time",
        "old_pred_x",
        "old_pred_y",
        "robust_pred_x",
        "robust_pred_y",
        "actual_x",
        "actual_y",
        "old_error_px",
        "robust_error_px",
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
        writer.writerows(rows)

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------

    print("\n평가 샘플 수:", len(rows))

    if not rows:

        print(
            "평가 가능한 trajectory가 없습니다."
        )

        return None

    old_mean = mean(old_errors)
    old_median = median(old_errors)

    robust_mean = mean(robust_errors)
    robust_median = median(robust_errors)

    print("\n[Trajectory Prediction Error]")

    print(
        "기존 방식 평균:",
        round(old_mean, 2),
        "px"
    )

    print(
        "기존 방식 중앙값:",
        round(old_median, 2),
        "px"
    )

    print(
        "Robust 방식 평균:",
        round(robust_mean, 2),
        "px"
    )

    print(
        "Robust 방식 중앙값:",
        round(robust_median, 2),
        "px"
    )

    print("\nCSV:", csv_path)

    return {
        "video": video_name,
        "samples": len(rows),
        "old_mean": old_mean,
        "old_median": old_median,
        "robust_mean": robust_mean,
        "robust_median": robust_median,
    }


# ============================================================
# 4. 전체 영상 평가
# ============================================================

summaries = []

for video_name in VIDEO_NAMES:

    try:

        result = evaluate_video(
            video_name
        )

        if result is not None:
            summaries.append(result)

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
# 5. 전체 결과
# ============================================================

print("\n\n===================================")
print("전체 Trajectory 평가 결과")
print("===================================")

for result in summaries:

    print(
        f"\n{result['video']}"
    )

    print(
        f"  Samples       : "
        f"{result['samples']}"
    )

    print(
        f"  Old Mean      : "
        f"{result['old_mean']:.2f} px"
    )

    print(
        f"  Old Median    : "
        f"{result['old_median']:.2f} px"
    )

    print(
        f"  Robust Mean   : "
        f"{result['robust_mean']:.2f} px"
    )

    print(
        f"  Robust Median : "
        f"{result['robust_median']:.2f} px"
    )

print("\n===================================")