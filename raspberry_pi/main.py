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


# =========================================================
# 경고 최소 유지시간
#
# CAUTION / DANGER 발생 후 이 시간 동안은
# SAFE 패킷을 ESP32로 보내지 않음
# =========================================================

WARNING_HOLD_MS = 500


async def main():
    sender = BLESender()

    camera = None

    # 마지막으로 CAUTION / DANGER가 발생한 시간
    last_warning_time = None

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
        # 2. 카메라 시작
        # =================================================

        print("[SYSTEM] Camera 시작")

        camera = Camera()
        camera.start()

        print("[SYSTEM] Camera started")


        # =================================================
        # 3. 실시간 스트리밍 서버 시작
        # =================================================

        stream_thread = threading.Thread(
            target=run_stream_server,
            daemon=True
        )

        stream_thread.start()

        print("[SYSTEM] Live stream server started")


        # =================================================
        # 4. 초기 스트림 프레임
        # =================================================

        print("[SYSTEM] 초기 카메라 화면 준비")

        frame = camera.get_frame()

        if frame is not None:
            update_frame(frame)

        print("[SYSTEM] 초기 카메라 화면 준비 완료")


        # =================================================
        # 5. BLE 연결
        # =================================================

        print("[SYSTEM] ESP32 BLE 연결 시작")

        await sender.connect_with_retry()

        print("[SYSTEM] ESP32 BLE 연결 완료")


        # =================================================
        # 6. 전체 통합 루프
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
            # 스트리밍 화면 갱신
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

                await asyncio.sleep(0)

                continue


            # ---------------------------------------------
            # 최종 Hazard 결과
            # ---------------------------------------------

            hazard = final_result

            print("[AI] result:", hazard)


            # =================================================
            # 경고 최소 유지시간 처리
            # =================================================

            current_time = now_ms()


            # hazard가 None이면 SAFE
            if hazard is None:
                risk = "SAFE"

            else:
                risk = hazard.get("risk", "SAFE")


            # ---------------------------------------------
            # CAUTION / DANGER
            #
            # 즉시 전송하고 마지막 경고 시간 갱신
            # ---------------------------------------------

            if risk in ("CAUTION", "DANGER"):
                last_warning_time = current_time


            # ---------------------------------------------
            # SAFE
            #
            # 마지막 경고 이후 500ms가 지나지 않았다면
            # SAFE 패킷을 보내지 않음
            # → 모터가 바로 꺼지는 것 방지
            # ---------------------------------------------

            elif risk == "SAFE":

                if last_warning_time is not None:

                    elapsed_from_warning = (
                        current_time - last_warning_time
                    )

                    if elapsed_from_warning < WARNING_HOLD_MS:
                        print(
                            f"[WARNING HOLD] SAFE 전송 보류 "
                            f"({elapsed_from_warning:.0f}"
                            f"/{WARNING_HOLD_MS} ms)"
                        )

                        await asyncio.sleep(0)

                        continue


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
            # 연결 유지 중이면 기존 연결 사용
            # 연결이 끊겼으면 BLESender 내부에서 재연결
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
        # 프로그램 종료 시 정리
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