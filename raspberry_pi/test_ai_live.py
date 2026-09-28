from pathlib import Path
import sys
import threading
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from camera import Camera
from latency import now_ms
from senseon_pipeline import FrameAnalyzer
from stream_server import update_frame, run_stream_server

# 모델 파일 경로
MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"


def main():
    print("[PATH] ROOT_DIR:", ROOT_DIR)
    print("[PATH] MODEL_PATH:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일 없음: {MODEL_PATH}")

    camera = Camera()
    analyzer = FrameAnalyzer(str(MODEL_PATH))

    # 브라우저 스트리밍 서버 실행
    threading.Thread(
        target=run_stream_server,
        daemon=True
    ).start()

    try:
        camera.start()
        print("[SYSTEM] Camera started")
        print("[SYSTEM] AI live streaming started")

        while True:
            # 1. 카메라 프레임 받기
            frame = camera.get_frame()

            # 2. timestamp
            timestamp = now_ms() / 1000.0

            # 3. AI 처리 + 박스 그리기
            final_result, state, annotated_frame = analyzer.process(
                frame,
                timestamp,
                annotate=True
            )

            # 4. 브라우저에 띄울 프레임 업데이트
            update_frame(annotated_frame)

            # 5. 터미널 로그
            if state == "READY":
                print("[AI] state=READY, result=", final_result)
            else:
                print("[AI] state=", state)

            # 너무 로그가 많으면 약간 쉬기
            time.sleep(0.03)

    finally:
        camera.stop()
        print("[SYSTEM] Camera stopped")


if __name__ == "__main__":
    main()