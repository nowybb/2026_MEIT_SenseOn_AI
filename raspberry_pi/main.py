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
        # 1. ESP32 BLE 연결
        # =================================================

        print("[SYSTEM] ESP32 BLE 연결 시작")

        await sender.connect_with_retry()

        print("[SYSTEM] ESP32 BLE 연결 완료")


        # =================================================
        # 2. AI 모델 로드
        # =================================================

        print("[SYSTEM] AI 모델 로드 시작")

        analyzer = FrameAnalyzer(
            str(MODEL_PATH)
        )

        print("[SYSTEM] AI 모델 로드 완료")


        # =================================================
        # 3. 카메라 시작
        # =================================================

        camera = Camera()
        camera.start()

        print("[SYSTEM] Camera started")


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
            # 카메라 프레임 획득
            # ---------------------------------------------

            frame = camera.get_frame()

            if frame is None:
                print("[CAMERA] frame is None")
                await asyncio.sleep(0.01)
                continue


            # ---------------------------------------------
            # AI 처리용 timestamp
            # ---------------------------------------------

            timestamp = now_ms() / 1000.0


            # ---------------------------------------------
            # AI 분석
            # ---------------------------------------------

            final_result, state, annotated_frame = analyzer.process(
                frame,
                timestamp,
                annotate=True
            )


            # AI 판단 완료 시점
            ai_result_time = now_ms()


            # ---------------------------------------------
            # 브라우저 스트리밍
            # ---------------------------------------------

            if annotated_frame is not None:
                update_frame(annotated_frame)

            else:
                update_frame(frame)


            # ---------------------------------------------
            # 아직 AI 판단 불가능
            # ---------------------------------------------

            if state != "READY":
                print(f"[AI] state={state}")

                await asyncio.sleep(0)
                continue


            # ---------------------------------------------
            # 최종 Hazard 결과
            # ---------------------------------------------

            hazard = final_result

            print("[AI] result:", hazard)


            # ---------------------------------------------
            # BLE Packet 생성
            # ---------------------------------------------

            packet = encode_hazard(hazard)

            print("[BLE] packet:", packet)


            # ---------------------------------------------
            # 이전 ACK 초기화
            # ---------------------------------------------

            sender.clear_ack()


            # ---------------------------------------------
            # ESP32 전송
            #
            # 연결이 살아 있으면 기존 연결 그대로 사용
            # 연결이 끊겨 있으면 BLESender 내부에서 재연결
            # ---------------------------------------------

            send_success = await sender.send(packet)

            if not send_success:
                print("[BLE] 전송 실패")

                await asyncio.sleep(0.01)
                continue


            # ---------------------------------------------
            # ESP32 ACK 대기
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
            # CSV 로그 저장
            # ---------------------------------------------

            save_log(
                hazard,
                e2e_latency_ms=e2e_latency
            )


            # 다른 asyncio 작업에게 실행 기회 제공
            await asyncio.sleep(0)


    except KeyboardInterrupt:
        print()
        print("[SYSTEM] 사용자 종료 요청")


    except asyncio.CancelledError:
        print()
        print("[SYSTEM] Task cancelled")
        raise


    except Exception as e:
        print()
        print(
            f"[SYSTEM] 예외 발생: "
            f"{type(e).__name__}: {repr(e)}"
        )


    finally:
        # =================================================
        # 프로그램이 실제로 종료될 때만 실행
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


        # BLE 연결 종료
        try:
            await sender.disconnect()

        except Exception as e:
            print(
                f"[SYSTEM] BLE 종료 오류: "
                f"{type(e).__name__}: {repr(e)}"
            )


        print("[SYSTEM] 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())