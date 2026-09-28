import cv2
from pathlib import Path

VIDEO_DIR = Path("toy_car_dataset/raw/videos")
OUTPUT_DIR = Path("toy_car_dataset/frames")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_INTERVAL = 10

for video_path in VIDEO_DIR.glob("*"):

    if video_path.suffix.lower() not in [".mp4", ".mov", ".avi"]:
        continue

    cap = cv2.VideoCapture(str(video_path))

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:

            output_path = OUTPUT_DIR / (
                f"{video_path.stem}_{saved_idx:04d}.jpg"
            )

            cv2.imwrite(str(output_path), frame)

            saved_idx += 1

        frame_idx += 1

    cap.release()

    print(
        f"{video_path.name}: "
        f"{saved_idx} frames saved"
    )