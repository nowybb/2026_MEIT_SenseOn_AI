import cv2
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from senseon_pipeline import FrameAnalyzer


MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"
VIDEO_PATH = ROOT_DIR / "test_videos" / "Test_DANGER_640.mp4"


def main():
    print("[SYSTEM] AI 모델 로딩...")
    analyzer = FrameAnalyzer(str(MODEL_PATH))
    print("[SYSTEM] AI 모델 로딩 완료")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("[ERROR] 영상을 열 수 없습니다:", VIDEO_PATH)
        return

    total_frames = 0
    total_ai_time = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        start = time.perf_counter()

        final_result, state, annotated_frame = analyzer.process(
            frame,
            timestamp,
            annotate=False
        )

        elapsed = time.perf_counter() - start

        total_ai_time += elapsed
        total_frames += 1

        current_fps = 1.0 / elapsed if elapsed > 0 else 0

        print(
            f"Frame {total_frames} | "
            f"AI time: {elapsed * 1000:.1f} ms | "
            f"AI FPS: {current_fps:.2f}"
        )

    cap.release()

    if total_frames > 0:
        avg_ai_time = total_ai_time / total_frames
        avg_fps = total_frames / total_ai_time

        print()
        print("========== RESULT ==========")
        print(f"Processed frames : {total_frames}")
        print(f"Total AI time    : {total_ai_time:.2f} s")
        print(f"Average AI time  : {avg_ai_time * 1000:.1f} ms/frame")
        print(f"Average AI FPS   : {avg_fps:.2f}")
        print("============================")


if __name__ == "__main__":
    main()