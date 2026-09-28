import asyncio
import threading
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from stream_server import update_frame, run_stream_server
from protocol import encode_hazard
from ble_sender import BLESender
from latency import now_ms, calc_latency_ms
from logger import save_log
from senseon_pipeline import FrameAnalyzer


MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"
VIDEO_PATH = ROOT_DIR / "test_videos" / "Test_DANGER.mp4"


async def main():

    sender = BLESender()

    cap = None
    ble_enabled = False

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
        # 4. BLE 연결 시도
        # =================================================

        print("[SYSTEM] ESP32 BLE 연결 시도")

        try:
            await asyncio.wait_for(
                sender.connect_with_retry(),
                timeout=10.0
            )

            ble_enabled = True

            print("[SYSTEM] ESP32 BLE 연결 완료")

        except Exception as e:

            ble_enabled = False

            print(
                "[SYSTEM] ESP32 연결 실패"
            )

            print(
                "[SYSTEM] AI + 영상 모드로 계속 실행"
            )

            print(
                f"[BLE ERROR] {e}"
            )


        # =================================================
        # 5. 테스트 영상 루프
        # =================================================

        while True:

            ret, frame = cap.read()

            # 영상 끝나면 처음부터
            if not ret:

                print(
                    "[SYSTEM] 영상 끝 -> 처음부터 다시 재생"
                )

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                continue


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
            # ESP32 없으면 여기까지만
            # =================================================

            if not ble_enabled:

                continue


            # =================================================
            # BLE 패킷 생성
            # =================================================

            packet = encode_hazard(
                hazard
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

                continue


            # =================================================
            # ACK 대기
            # =================================================

            ack_received = await sender.wait_for_ack()

            if not ack_received:

                print(
                    "[BLE] ACK 수신 실패"
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
            # 로그 저장
            # =================================================

            save_log(
                hazard,
                e2e_latency_ms=e2e_latency
            )


    except KeyboardInterrupt:

        print()
        print("[SYSTEM] 사용자 종료")


    finally:

        if cap is not None:
            cap.release()

        if ble_enabled:
            await sender.disconnect()

        print("[SYSTEM] 종료 완료")


if __name__ == "__main__":
    asyncio.run(
        main()
    )