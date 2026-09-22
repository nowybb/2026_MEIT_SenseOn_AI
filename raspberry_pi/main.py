# 1. Raspberry Pi 카메라 입력
# 2. AI 코드 통합
# 3. Raspberry Pi -> ESP32 BLE 통신
# 4. 통신 안정성 처리
# 5. End-to-End Latency 측정
# 6. 최종 통합 실행
# 통합 후 camera.py 호출

import asyncio

from camera import Camera
from protocol import encode_hazard
from ble_sender import BLESender
from latency import now_ms, calc_latency_ms
from logger import save_log
from senseon_pipeline import FrameAnalyzer


async def main():
    sender = BLESender()
    analyzer = FrameAnalyzer() # ()안에 모델 경로 작성해야 함.
    camera = Camera()

    try:
        # ESP32 연결
        await sender.connect_with_retry()

        # 카메라 시작
        camera.start()

        while True:
            # 1. 프레임 입력
            frame = camera.get_frame()

            # 2. AI 처리
            timestamp = now_ms() / 1000.0

            final_result, state, annotated_frame = analyzer.process(
                frame,
                timestamp
            )

            # 아직 AI가 판단 가능한 상태가 아니면 다음 프레임
            if state != "READY":
                print(f"AI 상태: {state}")
                continue

            hazard = final_result

            print("AI 결과:", hazard)

            # 3. BLE 패킷 생성
            packet = encode_hazard(hazard)

            print("전송 패킷:", packet)

            # 4. 이전 ACK 초기화
            sender.clear_ack()

            # 5. AI 판단 완료 시점
            ai_result_time = now_ms()

            # 6. BLE 전송
            send_success = await sender.send(packet)

            if not send_success:
                print("BLE 전송 실패")
                continue

            # 7. ACK 대기
            ack_received = await sender.wait_for_ack()

            if not ack_received:
                print("ACK 수신 실패")
                continue

            # 8. E2E latency 계산
            ack_time = now_ms()

            e2e_latency = calc_latency_ms(
                ai_result_time,
                ack_time
            )

            print(
                f"End-to-End Latency: "
                f"{e2e_latency:.3f} ms"
            )

            # 9. 로그 저장
            save_log(
                hazard,
                e2e_latency_ms=e2e_latency
            )

    finally:
        camera.stop()
        await sender.disconnect()


if __name__ == "__main__":
    asyncio.run(main())