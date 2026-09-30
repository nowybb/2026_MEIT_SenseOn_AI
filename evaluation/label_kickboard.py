from pathlib import Path
import csv
import cv2

# =========================
# 설정
# =========================

VIDEO_DIR = Path("ai1/videos")
OUTPUT_CSV = Path("evaluation/ground_truth/kickboard_events.csv")

VIDEO_NAMES = [
    "test6.mp4",
    "test7.mp4",
    "test8.mp4",
    "test9.mp4",
    "test10.mp4",
    "test11.mp4",
    "test12.mp4",
    "test13.mp4",
    "test14.mp4",
]

# 몇 초 간격으로 확인할지
STEP_SECONDS = 0.2


def load_existing():
    """기존 라벨이 있으면 불러온다."""
    data = {}

    if not OUTPUT_CSV.exists():
        return data

    with OUTPUT_CSV.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (row["video"], int(row["frame"]))
            data[key] = int(row["kickboard"])

    return data


def save_labels(labels):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for (video, frame), kickboard in labels.items():
        rows.append({
            "video": video,
            "frame": frame,
            "kickboard": kickboard,
        })

    rows.sort(key=lambda x: (x["video"], x["frame"]))

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frame",
                "kickboard",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    labels = load_existing()

    print("=" * 60)
    print("킥보드 Ground Truth 라벨링")
    print("=" * 60)
    print()
    print("키")
    print("  1 : 킥보드 있음")
    print("  0 : 킥보드 없음")
    print("  S : 건너뛰기")
    print("  Q : 저장 후 종료")
    print()

    for video_name in VIDEO_NAMES:

        video_path = VIDEO_DIR / video_name

        if not video_path.exists():
            print(f"[없음] {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0:
            print(f"[FPS 오류] {video_name}")
            cap.release()
            continue

        step_frames = max(
            1,
            int(fps * STEP_SECONDS)
        )

        print()
        print("=" * 60)
        print(video_name)
        print(
            f"FPS={fps:.2f} / "
            f"frames={total_frames} / "
            f"interval={STEP_SECONDS}s"
        )
        print("=" * 60)

        frame_number = 0

        while frame_number < total_frames:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            ok, frame = cap.read()

            if not ok:
                break

            timestamp = frame_number / fps

            key = (
                video_name,
                frame_number
            )

            # 기존 라벨 표시
            existing = labels.get(key)

            display = frame.copy()

            text = (
                f"{video_name} | "
                f"{timestamp:.2f}s | "
                f"frame {frame_number}"
            )

            cv2.putText(
                display,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if existing is not None:

                cv2.putText(
                    display,
                    f"Existing label: {existing}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                "Kickboard Ground Truth",
                display
            )

            pressed = cv2.waitKey(0) & 0xFF

            # 1
            if pressed == ord("1"):

                labels[key] = 1
                print(
                    f"{video_name} "
                    f"{timestamp:.2f}s -> kickboard"
                )

            # 0
            elif pressed == ord("0"):

                labels[key] = 0
                print(
                    f"{video_name} "
                    f"{timestamp:.2f}s -> none"
                )

            # q
            elif pressed in (
                ord("q"),
                ord("Q")
            ):

                save_labels(labels)

                cap.release()
                cv2.destroyAllWindows()

                print()
                print(
                    f"저장 완료: {OUTPUT_CSV}"
                )

                return

            # s는 아무것도 저장하지 않음

            frame_number += step_frames

        cap.release()

        # 영상 하나 끝날 때마다 저장
        save_labels(labels)

        print(
            f"{video_name} 완료 / 자동 저장"
        )

    cv2.destroyAllWindows()

    save_labels(labels)

    print()
    print("=" * 60)
    print("전체 라벨링 완료")
    print(f"저장: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()