import asyncio
import threading
import sys
import traceback
from pathlib import Path

from camera import Camera
from protocol import encode_hazard
from ble_sender import BLESender
from latency import now_ms, calc_latency_ms
from logger import save_log
from stream_server import update_frame, run_stream_server


# =========================================================
# AI 프로젝트 경로 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from senseon_pipeline import FrameAnalyzer


MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"


async def main():
    sender = BLESender()

    camera = None

    try:
        # =================================================
        # 1. AI 모델 먼저 로드
        # =================================================

        print("[SYSTEM] AI 모델 로드 시작")

        analyzer = FrameAnalyzer(
            str(MODEL_PATH)
        )

        print("[SYSTEM] AI 모델 로드 완료")


        # =================================================
        # 2. 카메라 먼저 시작
        # =================================================

        print("[SYSTEM] Camera 시작")

        camera = Camera()
        camera.start()

        print("[SYSTEM] Camera started")


        # =================================================
        # 3. 무거운 초기화가 끝난 뒤 BLE 연결
        # =================================================

        print("[SYSTEM] ESP32 BLE 연결 시작")

        await sender.connect_with_retry()

        print("[SYSTEM] ESP32 BLE 연결 완료")


        # =================================================
        # 4. 실시간 스트리밍 시작
        # =================================================

        stream_thread = threading.Thread(
            target=run_stream_server,
            daemon=True
        )

        stream_thread.start()

        print("[SYSTEM] Live stream started")


        # =================================================
        # 5. 전체 통합 루프
        # =================================================

        while True:

            # ---------------------------------------------
            # 카메라 프레임
            # ---------------------------------------------

            frame = camera.get_frame()

            if frame is None:
                await asyncio.sleep(0.01)
                continue


            timestamp = now_ms() / 1000.0


            # ---------------------------------------------
            # AI 추론
            #
            # 중요:
            # analyzer.process()가 asyncio 이벤트 루프를
            # 막지 않도록 별도 thread에서 실행
            # ---------------------------------------------

            final_result, state, annotated_frame = (
                await asyncio.to_thread(
                    analyzer.process,
                    frame,
                    timestamp,
                    annotate=True
                )
            )


            # AI 판단 완료 시점
            ai_result_time = now_ms()


            # ---------------------------------------------
            # 스트리밍
            # ---------------------------------------------

            if annotated_frame is not None:
                update_frame(annotated_frame)

            else:
                update_frame(frame)


            # ---------------------------------------------
            # AI 상태 확인
            # ---------------------------------------------

            if state != "READY":
                print(f"[AI] state={state}")

                # 이벤트 루프에 실행권 반환
                await asyncio.sleep(0)

                continue


            # ---------------------------------------------
            # AI 최종 결과
            # ---------------------------------------------

            hazard = final_result

            print("[AI] result:", hazard)


            # ---------------------------------------------
            # BLE Packet
            # ---------------------------------------------

            packet = encode_hazard(hazard)

            print("[BLE] packet:", packet)


            # ---------------------------------------------
            # ACK 초기화
            # ---------------------------------------------

            sender.clear_ack()


            # ---------------------------------------------
            # ESP32 전송
            #
            # 연결되어 있으면 기존 연결 사용
            # 끊어졌으면 BLESender 내부에서 재연결
            # ---------------------------------------------

            send_success = await sender.send(packet)

            if not send_success:
                print("[BLE] 전송 실패")

                await asyncio.sleep(0.01)

                continue


            # ---------------------------------------------
            # ACK 대기
            # ---------------------------------------------

            ack_received = await sender.wait_for_ack()

            if not ack_received:
                print("[BLE] ACK 수신 실패")

                await asyncio.sleep(0.01)

                continue


            # ---------------------------------------------
            # End-to-End Latency
            # ---------------------------------------------

            ack_time = now_ms()

            e2e_latency = calc_latency_ms(
                ai_result_time,
                ack_time
            )

            print(
                f"[LATENCY] End-to-End: "
                f"{e2e_latency:.3f} ms"
            )


            # ---------------------------------------------
            # CSV 저장
            # ---------------------------------------------

            save_log(
                hazard,
                e2e_latency_ms=e2e_latency
            )


            # asyncio 이벤트 루프에 실행권 반환
            await asyncio.sleep(0)


    except asyncio.CancelledError:
        print("[SYSTEM] Task cancelled")
        raise


    except Exception as e:
        print()
        print(
            f"[SYSTEM] 예외 발생: "
            f"{type(e).__name__}: {repr(e)}"
        )

        traceback.print_exc()


    finally:
        # =================================================
        # 프로그램이 실제로 종료될 때만 BLE 연결 종료
        # =================================================

        print("[SYSTEM] 종료 처리 시작")


        # 카메라 종료
        if camera is not None:
            try:
                camera.stop()

                print("[SYSTEM] Camera stopped")

            except Exception as e:
                print(
                    f"[SYSTEM] Camera 종료 오류: "
                    f"{type(e).__name__}: {repr(e)}"
                )


        # BLE 종료
        try:
            await sender.disconnect()

        except Exception as e:
            print(
                f"[SYSTEM] BLE 종료 오류: "
                f"{type(e).__name__}: {repr(e)}"
            )


        print("[SYSTEM] 종료 완료")


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print()
        print("[SYSTEM] 사용자 종료")