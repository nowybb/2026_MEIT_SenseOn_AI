import csv
import os
import cv2


VIDEO_PATH = "ai1/videos/test_receding.mp4"
OUTPUT_PATH = "evaluation/ground_truth/events.csv"

video_name = os.path.basename(VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"영상을 열 수 없습니다: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

# 영상 전체 길이 계산
total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

video_duration = (
    total_frames / fps
)

event_start = None
event_id = 1


def save_event(start_sec, end_sec):
    global event_id

    with open(
        OUTPUT_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            video_name,
            event_id,
            round(start_sec, 2),
            round(end_sec, 2),
            1,
            "위험 상황",
        ])

    print(
        f"[GT 저장] "
        f"{video_name} / event {event_id} / "
        f"{start_sec:.2f}s ~ {end_sec:.2f}s"
    )

    event_id += 1


print(
    f"[영상] {video_name} / "
    f"길이 {video_duration:.2f}s"
)

print(
    "[조작] "
    "S=위험 시작 / "
    "E=위험 종료 / "
    "SPACE=일시정지 / "
    "Q=종료"
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_frame = int(
        cap.get(cv2.CAP_PROP_POS_FRAMES)
    )

    current_sec = (
        current_frame / fps
    )

    status = (
        "DANGER LABELING"
        if event_start is not None
        else "SAFE"
    )

    cv2.putText(
        frame,
        f"{current_sec:.2f}s",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255)
        if event_start is not None
        else (0, 255, 0),
        2,
    )

    cv2.imshow(
        "Ground Truth Labeler",
        frame,
    )

    key = cv2.waitKey(
        max(1, int(1000 / fps))
    ) & 0xFF

    # S = 위험 구간 시작
    if key == ord("s"):

        if event_start is None:

            event_start = current_sec

            print(
                f"[위험 시작] "
                f"{event_start:.2f}s"
            )

    # E = 위험 구간 종료
    elif key == ord("e"):

        if event_start is not None:

            save_event(
                event_start,
                current_sec,
            )

            event_start = None

    # SPACE = 일시정지
    elif key == ord(" "):

        print(
            f"[PAUSE] "
            f"{current_sec:.2f}s"
        )

        while True:

            pause_key = (
                cv2.waitKey(0)
                & 0xFF
            )

            if pause_key == ord(" "):
                break

            if pause_key == ord("q"):

                cap.release()
                cv2.destroyAllWindows()

                raise SystemExit

    # Q = 종료
    elif key == ord("q"):
        break


# ============================================================
# 영상 끝까지 위험 상태가 유지된 경우 자동 저장
# ============================================================

if event_start is not None:

    save_event(
        event_start,
        video_duration,
    )

    print(
        "[GT] 위험 상태가 영상 끝까지 유지되어 "
        "마지막 프레임까지 자동 저장했습니다."
    )


cap.release()
cv2.destroyAllWindows()