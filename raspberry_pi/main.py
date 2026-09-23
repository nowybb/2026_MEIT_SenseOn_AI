import asyncio
import threading
import sys
from pathlib import Path

from camera import Camera
from protocol import encode_hazard
from ble_sender import BLESender
from latency import now_ms, calc_latency_ms
from logger import save_log
from stream_server import update_frame, run_stream_server


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from senseon_pipeline import FrameAnalyzer


MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"


async def main():
    sender = BLESender()

    camera = None

    try:
        # =========================
        # 1. BLE 먼저 연결
        # =========================
        print("[SYSTEM] ESP32 BLE 연결 시작")

        await sender.connect_with_retry()

        print("[SYSTEM] ESP32 BLE 연결 완료")


        # =========================
        # 2. AI 모델 로드
        # =========================
        print("[SYSTEM] AI 모델 로드 시작")

        analyzer = FrameAnalyzer(str(MODEL_PATH))

        print("[SYSTEM] AI 모델 로드 완료")


        # =========================
        # 3. 카메라 생성 및 시작
        # =========================
        camera = Camera()
        camera.start()

        print("[SYSTEM] Camera started")


        # =========================
        # 4. 실시간 스트리밍 시작
        # =========================
        threading.Thread(
            target=run_stream_server,
            daemon=True
        ).start()

        print("[SYSTEM] Live stream started")


        # =========================
        # 5. 전체 통합 루프
        # =========================
        while True:

            # 카메라 프레임
            frame = camera.get_frame()

            timestamp = now_ms() / 1000.0


            # AI 처리
            final_result, state, annotated_frame = analyzer.process(
                frame,
                timestamp,
                annotate=True
            )

            # AI 판단 완료 시점
            ai_result_time = now_ms()


            # 브라우저 화면
            if annotated_frame is not None:
                update_frame(annotated_frame)
            else:
                update_frame(frame)


            # 아직 판단 불가능
            if state != "READY":
                print(f"[AI] state={state}")
                continue


            hazard = final_result

            print("[AI] result:", hazard)


            # BLE 패킷
            packet = encode_hazard(hazard)

            print("[BLE] packet:", packet)


            # 이전 ACK 초기화
            sender.clear_ack()


            # ESP32 전송
            send_success = await sender.send(packet)

            if not send_success:
                print("[BLE] 전송 실패")
                continue


            # ACK 대기
            ack_received = await sender.wait_for_ack()

            if not ack_received:
                print("[BLE] ACK 수신 실패")
                continue


            # E2E Latency
            ack_time = now_ms()

            e2e_latency = calc_latency_ms(
                ai_result_time,
                ack_time
            )

            print(
                f"[LATENCY] End-to-End: "
                f"{e2e_latency:.3f} ms"
            )


            # CSV
            save_log(
                hazard,
                e2e_latency_ms=e2e_latency
            )


    finally:
        if camera is not None:
            camera.stop()

        await sender.disconnect()

        print("[SYSTEM] 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())