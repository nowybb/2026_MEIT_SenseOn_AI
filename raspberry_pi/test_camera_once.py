from camera import Camera
from senseon_pipeline import FrameAnalyzer
from latency import now_ms


MODEL_PATH = "모델파일경로.pt"   # 나중에 실제 경로로 수정


def main():
    camera = Camera()
    analyzer = FrameAnalyzer(MODEL_PATH)

    try:
        # 1. 카메라 시작
        camera.start()

        # 2. 프레임 1장 받기
        frame = camera.get_frame()

        print("[CAMERA] frame shape:", frame.shape)
        print("[CAMERA] frame type:", type(frame))

        # 3. AI 입력 timestamp
        timestamp = now_ms() / 1000.0

        # 4. AI 1회 처리
        final_result, state, annotated_frame = analyzer.process(
            frame,
            timestamp,
            annotate=True
        )

        print("[AI] state:", state)
        print("[AI] final_result:", final_result)
        print("[AI] annotated_frame shape:", annotated_frame.shape)

    finally:
        camera.stop()


if __name__ == "__main__":
    main()