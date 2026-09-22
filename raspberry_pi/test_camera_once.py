from pathlib import Path
import sys

# GitHub 프로젝트 최상위 폴더
ROOT_DIR = Path(__file__).resolve().parent.parent

# senseon_pipeline.py / ai2 등을 import할 수 있게 최상위 폴더 추가
sys.path.insert(0, str(ROOT_DIR))

from camera import Camera
from senseon_pipeline import FrameAnalyzer
from latency import now_ms


# AI 모델 경로
MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"


def main():
    print("[PATH] ROOT_DIR:", ROOT_DIR)
    print("[PATH] MODEL_PATH:", MODEL_PATH)

    # 모델 파일 존재 여부 먼저 확인
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO 모델 파일을 찾을 수 없음: {MODEL_PATH}"
        )

    camera = Camera()
    analyzer = FrameAnalyzer(str(MODEL_PATH))

    try:
        # 1. 카메라 시작
        camera.start()

        # 2. 프레임 1장 받기
        frame = camera.get_frame()

        print("[CAMERA] frame shape:", frame.shape)
        print("[CAMERA] frame type:", type(frame))

        # 3. AI 입력 timestamp (초 단위)
        timestamp = now_ms() / 1000.0

        # 4. AI 1회 처리
        final_result, state, annotated_frame = analyzer.process(
            frame,
            timestamp,
            annotate=True
        )

        print("[AI] state:", state)
        print("[AI] final_result:", final_result)

        if annotated_frame is not None:
            print("[AI] annotated_frame shape:", annotated_frame.shape)
        else:
            print("[AI] annotated_frame: None")

    finally:
        camera.stop()


if __name__ == "__main__":
    main()