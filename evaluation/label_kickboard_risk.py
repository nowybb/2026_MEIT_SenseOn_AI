from pathlib import Path
import cv2
import csv

VIDEO_DIR = Path("ai1/videos")
OUTPUT = Path("evaluation/ground_truth/kickboard_risk_events.csv")

VIDEOS = [f"test{i}.mp4" for i in range(6, 15)]

STEP_SEC = 0.2

# S = SAFE
# D = DANGER
KEY_MAP = {
    ord("s"): "SAFE",
    ord("d"): "DANGER",
}


def save(rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "timestamp",
                "gt_level",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():

    rows = []

    print("=" * 60)
    print("KICKBOARD RISK GROUND TRUTH")
    print("=" * 60)
    print("S = SAFE")
    print("D = DANGER")
    print("Q = 저장 후 종료")
    print()

    for video_name in VIDEOS:

        path = VIDEO_DIR / video_name

        if not path.exists():
            print(f"[없음] {path}")
            continue

        cap = cv2.VideoCapture(str(path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            print(f"[FPS 오류] {video_name}")
            cap.release()
            continue

        step_frames = max(1, int(fps * STEP_SEC))

        print()
        print("=" * 60)
        print(video_name)
        print("=" * 60)

        frame_no = 0

        while frame_no < total_frames:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_no
            )

            ok, frame = cap.read()

            if not ok:
                break

            timestamp = frame_no / fps

            display = frame.copy()

            cv2.putText(
                display,
                f"{video_name} | {timestamp:.2f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display,
                "S=SAFE  D=DANGER  Q=QUIT",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Kickboard Risk Ground Truth",
                display
            )

            key = cv2.waitKey(0) & 0xFF

            # 종료
            if key in (ord("q"), ord("Q")):

                save(rows)

                cap.release()
                cv2.destroyAllWindows()

                print()
                print(f"저장 완료: {OUTPUT}")

                return

            # 대문자도 처리
            if key < 128:
                key = ord(chr(key).lower())

            if key in KEY_MAP:

                level = KEY_MAP[key]

                rows.append({
                    "video": video_name,
                    "timestamp": f"{timestamp:.2f}",
                    "gt_level": level,
                })

                print(
                    f"{video_name} "
                    f"{timestamp:.2f}s -> {level}"
                )

            frame_no += step_frames

        cap.release()

        # 영상 하나 끝날 때마다 자동 저장
        save(rows)

        print(f"{video_name} 완료 / 자동 저장")

    cv2.destroyAllWindows()

    save(rows)

    print()
    print("=" * 60)
    print("전체 라벨링 완료")
    print(f"저장: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()