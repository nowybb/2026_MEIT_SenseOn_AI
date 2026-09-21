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
)
from ai2.ttc import calculate_visual_ttc_robust
from ai2.risk import (
    determine_risk_level,
    RiskStabilizer,
)


# ============================================================
# 1. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AI1_DIR = BASE_DIR / "ai1"
VIDEO_DIR = AI1_DIR / "videos"
OUTPUT_DIR = AI1_DIR / "outputs" / "risk_evaluation"

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

RELEASE_FRAMES_LIST = [
    1,
    3,
    5,
    10,
]

RISK_LEVELS = [
    "SAFE",
    "CAUTION",
    "DANGER",
]


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
    print("Risk 평가 시작:", video_name)
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

    print("FPS:", fps)

    # --------------------------------------------------------
    # Collision Zone
    # --------------------------------------------------------

    collision_zone = create_collision_zone(
        frame_width,
        frame_height,
    )

    # --------------------------------------------------------
    # 모델
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
    # History
    # --------------------------------------------------------

    track_history = TrackHistory(
        max_history=HISTORY_SIZE
    )

    # release_frames별 독립 Stabilizer
    stabilizers = {
        release_frames: RiskStabilizer(
            release_frames=release_frames
        )
        for release_frames
        in RELEASE_FRAMES_LIST
    }

    # track별 이전 안정화 상태
    previous_states = {
        release_frames: {}
        for release_frames
        in RELEASE_FRAMES_LIST
    }

    # 통계
    statistics = {}

    for release_frames in RELEASE_FRAMES_LIST:

        statistics[release_frames] = {
            "samples": 0,
            "safe": 0,
            "caution": 0,
            "danger": 0,
            "state_changes": 0,
            "danger_frames": 0,
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
            # Trajectory
            #
            # 기존 baseline trajectory 유지.
            # 이번 실험에서는 Risk Stabilizer만 비교한다.
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
            # Collision
            #
            # 기존 center collision 유지.
            # ------------------------------------------------

            path_collision = (
                check_path_collision(
                    current_position,
                    predicted_position,
                    collision_zone,
                )
            )

            # ------------------------------------------------
            # TTC
            #
            # 앞 단계에서 안정성이 개선된 Robust TTC 사용.
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

            row = {
                "frame": frame_index,
                "timestamp": timestamp,
                "track_id": track_id,
                "class_name": class_name,
                "approaching": approaching,
                "path_collision": path_collision,
                "visual_ttc": visual_ttc,
                "raw_risk": raw_risk,
            }

            # ------------------------------------------------
            # release_frames별 비교
            # ------------------------------------------------

            for release_frames in RELEASE_FRAMES_LIST:

                stabilizer = stabilizers[
                    release_frames
                ]

                stable_risk = (
                    stabilizer.update(
                        track_id,
                        raw_risk,
                    )
                )

                stats = statistics[
                    release_frames
                ]

                stats["samples"] += 1

                stats[
                    stable_risk.lower()
                ] += 1

                if stable_risk == "DANGER":
                    stats[
                        "danger_frames"
                    ] += 1

                previous = (
                    previous_states[
                        release_frames
                    ].get(track_id)
                )

                changed = False

                if (
                    previous is not None
                    and previous != stable_risk
                ):

                    changed = True

                    stats[
                        "state_changes"
                    ] += 1

                previous_states[
                    release_frames
                ][track_id] = stable_risk

                row[
                    f"risk_release_{release_frames}"
                ] = stable_risk

                row[
                    f"changed_release_{release_frames}"
                ] = changed

            rows.append(
                row
            )

    # ========================================================
    # 4. CSV 저장
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_risk_evaluation.csv"
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
    ]

    for release_frames in RELEASE_FRAMES_LIST:

        fieldnames.append(
            f"risk_release_{release_frames}"
        )

        fieldnames.append(
            f"changed_release_{release_frames}"
        )

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

    for release_frames in RELEASE_FRAMES_LIST:

        stats = statistics[
            release_frames
        ]

        danger_seconds = (
            stats["danger_frames"]
            / fps
        )

        print(
            "\n-----------------------------------"
        )

        print(
            "Release Frames:",
            release_frames
        )

        print(
            "SAFE:",
            stats["safe"]
        )

        print(
            "CAUTION:",
            stats["caution"]
        )

        print(
            "DANGER:",
            stats["danger"]
        )

        print(
            "State Changes:",
            stats["state_changes"]
        )

        print(
            "DANGER Frames:",
            stats["danger_frames"]
        )

        print(
            "Approx. DANGER Seconds:",
            round(
                danger_seconds,
                2
            )
        )

    print(
        "\nCSV:",
        csv_path
    )

    return {
        "fps": fps,
        "statistics": statistics,
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
print("전체 Risk 평가 결과")
print("===================================")

for video_name, result in all_results.items():

    fps = result["fps"]

    print(
        f"\n{video_name}"
    )

    for release_frames in RELEASE_FRAMES_LIST:

        stats = result[
            "statistics"
        ][release_frames]

        danger_seconds = (
            stats["danger_frames"]
            / fps
        )

        print(
            f"  release={release_frames:2d}"
            f" | changes={stats['state_changes']:4d}"
            f" | SAFE={stats['safe']:4d}"
            f" | CAUTION={stats['caution']:4d}"
            f" | DANGER={stats['danger']:4d}"
            f" | danger_sec={danger_seconds:.2f}"
        )

print("\n===================================")