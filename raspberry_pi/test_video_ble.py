import asyncio
import threading
import sys
from pathlib import Path

import cv2

from stream_server import update_frame, run_stream_server
from protocol import encode_hazard
from ble_sender import BLESender
from latency import now_ms, calc_latency_ms
from logger import save_log


# =========================================================
# 프로젝트 경로
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from senseon_pipeline import FrameAnalyzer


# =========================================================
# 설정
# =========================================================

MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"

# 테스트 영상
VIDEO_PATH = ROOT_DIR / "test_videos" / "Test_DANGER.mp4"


async def main():
    sender = BLESender()

    cap = None

    try:
        # =================================================
        # 1. AI 모델 로드
        # =================================================

        print("[SYSTEM] AI 모델 로드 시작")

        analyzer = FrameAnalyzer(
            str(MODEL_PATH)
        )

        print("[SYSTEM] AI 모델 로드 완료")


        # =================================================
        # 2. 영상 파일 열기
        # =================================================

        cap = cv2.VideoCapture(
            str(VIDEO_PATH)
        )

        if not cap.isOpened():
            print(
                f"[ERROR] 영상 파일을 열 수 없습니다: "
                f"{VIDEO_PATH}"
            )
            return


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30.0

        frame_delay = 1.0 / fps

        print(
            f"[SYSTEM] 테스트 영상 로드 완료 "
            f"({fps:.2f} FPS)"
        )


        # =================================================
        # 3. 스트리밍 서버 시작
        # =================================================

        stream_thread = threading.Thread(
            target=run_stream_server,
            daemon=True
        )

        stream_thread.start()

        print("[SYSTEM] Live stream server started")


        # =================================================
        # 4. BLE 연결
        # =================================================

        print("[SYSTEM] ESP32 BLE 연결 시작")

        await sender.connect_with_retry()

        print("[SYSTEM] ESP32 BLE 연결 완료")


        # =================================================
        # 5. 테스트 영상 루프
        # =================================================

        while True:

            ret, frame = cap.read()

            # 영상 끝
            if not ret:
                print("[SYSTEM] 영상 끝 -> 처음부터 다시 재생")

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                continue


            # 영상 기준 timestamp
            timestamp = (
                cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )
                / 1000.0
            )


            # =================================================
            # AI 분석
            # =================================================

            final_result, state, annotated_frame = (
                await asyncio.to_thread(
                    analyzer.process,
                    frame,
                    timestamp,
                    annotate=True
                )
            )


            ai_result_time = now_ms()


            # =================================================
            # 브라우저 스트리밍
            # =================================================

            if annotated_frame is not None:
                update_frame(
                    annotated_frame
                )

            else:
                update_frame(
                    frame
                )


            # =================================================
            # AI 준비 안 된 상태
            # =================================================

            if state != "READY":
                print(
                    f"[AI] state={state}"
                )

                await asyncio.sleep(
                    frame_delay
                )

                continue


            # =================================================
            # Hazard 결과
            # =================================================

            hazard = final_result

            print(
                "[AI] result:",
                hazard
            )


            # =================================================
            # BLE 패킷 생성
            # =================================================

            packet = encode_hazard(
                hazard
            )

            print(
                "[BLE] packet:",
                packet
            )


            sender.clear_ack()


            # =================================================
            # ESP32 전송
            # =================================================

            send_success = await sender.send(
                packet
            )

            if not send_success:
                print(
                    "[BLE] 전송 실패"
                )

                await asyncio.sleep(
                    frame_delay
                )

                continue


            # =================================================
            # ACK 대기
            # =================================================

            ack_received = await sender.wait_for_ack()

            if not ack_received:
                print(
                    "[BLE] ACK 수신 실패"
                )

                await asyncio.sleep(
                    frame_delay
                )

                continue


            # =================================================
            # E2E Latency
            # =================================================

            ack_time = now_ms()

            e2e_latency = calc_latency_ms(
                ai_result_time,
                ack_time
            )

            print(
                f"[LATENCY] End-to-End: "
                f"{e2e_latency:.3f} ms"
            )


            # =================================================
            # CSV 로그 저장
            # =================================================

            save_log(
                hazard,
                e2e_latency_ms=e2e_latency
            )


            # =================================================
            # 원본 영상 FPS 기준 재생
            # =================================================

            await asyncio.sleep(
                frame_delay
            )


    except KeyboardInterrupt:
        print()
        print("[SYSTEM] 사용자 종료")


    finally:

        if cap is not None:
            cap.release()

        await sender.disconnect()

        print("[SYSTEM] 종료 완료")


if __name__ == "__main__":
    asyncio.run(
        main()
    )