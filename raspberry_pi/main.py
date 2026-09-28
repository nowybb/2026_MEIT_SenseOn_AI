import asyncio
import threading
import sys
import traceback
import time
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
# BLE 설정
# =========================================================

# 최대 BLE 전송 주기
# 200ms = 최대 5Hz
BLE_INTERVAL_MS = 200


# =========================================================
# BLE 비동기 전송
# =========================================================

async def send_hazard_ble(
    sender,
    hazard,
    ai_result_time
):

    try:

        packet = encode_hazard(hazard)

        sender.clear_ack()

        send_success = await sender.send(
            packet
        )

        if not send_success:

            print("[BLE] 전송 실패")
            return


        ack_received = await sender.wait_for_ack()

        if not ack_received:

            print("[BLE] ACK 수신 실패")
            return


        # ---------------------------------------------
        # ACK 수신 시점
        # ---------------------------------------------

        ack_time = now_ms()


        # ---------------------------------------------
        # End-to-End Latency
        # AI 판단 완료 -> ACK 수신
        # ---------------------------------------------

        e2e_latency = calc_latency_ms(
            ai_result_time,
            ack_time
        )


        print(
            f"[LATENCY] End-to-End: "
            f"{e2e_latency:.1f} ms"
        )


        # ---------------------------------------------
        # CSV 로그 저장
        # ---------------------------------------------

        save_log(
            hazard,
            e2e_latency_ms=e2e_latency
        )


    except Exception as e:

        print(
            f"[BLE ERROR] "
            f"{type(e).__name__}: {repr(e)}"
        )


# =========================================================
# MAIN
# =========================================================

async def main():

    sender = BLESender()

    camera = None

    capture_thread = None
    capture_stop = threading.Event()

    ble_task = None

    last_ble_time = 0


    # =====================================================
    # 최신 프레임 공유 변수
    # =====================================================

    frame_lock = threading.Lock()

    latest_frame = None
    latest_timestamp = 0.0
    latest_frame_id = 0


    # =====================================================
    # 카메라 캡처 스레드
    #
    # AI 처리속도와 관계없이 계속 카메라를 읽음
    # 이전 프레임을 저장하지 않고 최신 프레임만 유지
    # =====================================================

    def capture_loop():

        nonlocal latest_frame
        nonlocal latest_timestamp
        nonlocal latest_frame_id

        print(
            "[CAMERA] Latest-frame capture thread started"
        )


        while not capture_stop.is_set():

            try:

                frame = camera.get_frame()

                if frame is None:
                    continue


                timestamp = now_ms() / 1000.0


                with frame_lock:

                    # 최신 프레임으로 계속 덮어쓰기
                    latest_frame = frame
                    latest_timestamp = timestamp

                    latest_frame_id += 1


            except Exception as e:

                if not capture_stop.is_set():

                    print(
                        f"[CAMERA ERROR] "
                        f"{type(e).__name__}: {repr(e)}"
                    )

                break


        print(
            "[CAMERA] Capture thread stopped"
        )


    try:

        # =================================================
        # 1. AI 모델 로드
        # =================================================

        print(
            "[SYSTEM] AI 모델 로드 시작"
        )


        analyzer = FrameAnalyzer(
            str(MODEL_PATH)
        )


        print(
            "[SYSTEM] AI 모델 로드 완료"
        )


        # =================================================
        # 2. 카메라 시작
        # =================================================

        print(
            "[SYSTEM] Camera 시작"
        )


        camera = Camera()

        camera.start()


        print(
            "[SYSTEM] Camera started"
        )


        # =================================================
        # 3. 카메라 최신 프레임 캡처 스레드 시작
        # =================================================

        capture_thread = threading.Thread(
            target=capture_loop,
            daemon=True
        )

        capture_thread.start()


        # =================================================
        # 4. 실시간 스트리밍 서버 시작
        # =================================================

        stream_thread = threading.Thread(
            target=run_stream_server,
            daemon=True
        )

        stream_thread.start()


        print(
            "[SYSTEM] Live stream server started"
        )


        # =================================================
        # 5. 첫 카메라 프레임 기다리기
        # =================================================

        print(
            "[SYSTEM] 초기 카메라 화면 준비"
        )


        while True:

            with frame_lock:

                if latest_frame is not None:

                    initial_frame = (
                        latest_frame.copy()
                    )

                    break


            await asyncio.sleep(0.01)


        update_frame(
            initial_frame
        )


        print(
            "[SYSTEM] 초기 카메라 화면 준비 완료"
        )


        # =================================================
        # 6. BLE 연결
        # =================================================

        print(
            "[SYSTEM] ESP32 BLE 연결 시작"
        )


        await sender.connect_with_retry()


        print(
            "[SYSTEM] ESP32 BLE 연결 완료"
        )


        # =================================================
        # 7. 통계값
        # =================================================

        processed_frames = 0

        last_processed_frame_id = 0

        total_ai_time = 0.0

        stats_start_time = time.monotonic()


        # =================================================
        # 8. AI 통합 루프
        #
        # 항상 "가장 최신 프레임"만 분석
        # =================================================

        while True:

            # ---------------------------------------------
            # 최신 프레임 가져오기
            # ---------------------------------------------

            with frame_lock:

                current_frame_id = latest_frame_id


                # AI가 마지막으로 처리했던 프레임과
                # 같은 프레임이면 새 프레임 기다림
                if (
                    latest_frame is None
                    or current_frame_id
                    == last_processed_frame_id
                ):

                    frame = None

                else:

                    frame = latest_frame.copy()

                    timestamp = latest_timestamp


            if frame is None:

                await asyncio.sleep(0.001)

                continue


            # =================================================
            # Frame Drop 계산
            #
            # 예:
            # 마지막 처리 ID = 10
            # 현재 최신 ID = 17
            #
            # 11~16은 처리하지 않고 최신 17 처리
            # -> 6 frame drop
            # =================================================

            dropped_frames = (
                current_frame_id
                - last_processed_frame_id
                - 1
            )


            if (
                last_processed_frame_id > 0
                and dropped_frames > 0
            ):

                print(
                    f"[FRAME] "
                    f"Dropped {dropped_frames} frame(s)"
                )


            last_processed_frame_id = (
                current_frame_id
            )


            # =================================================
            # AI 추론
            # =================================================

            ai_start = time.perf_counter()


            final_result, state, annotated_frame = (
                await asyncio.to_thread(
                    analyzer.process,
                    frame,
                    timestamp,
                    annotate=True
                )
            )


            ai_elapsed = (
                time.perf_counter()
                - ai_start
            )


            total_ai_time += ai_elapsed

            processed_frames += 1


            # AI 판단 완료 시점
            ai_result_time = now_ms()


            # =================================================
            # 현재 AI 처리 성능 출력
            # =================================================

            current_ai_fps = (
                1.0 / ai_elapsed
                if ai_elapsed > 0
                else 0
            )


            print(
                f"[AI PERFORMANCE] "
                f"{ai_elapsed * 1000:.1f} ms/frame | "
                f"{current_ai_fps:.2f} FPS"
            )


            # =================================================
            # 스트리밍 화면 갱신
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
            # AI READY 확인
            # =================================================

            if state != "READY":

                print(
                    f"[AI] state={state}"
                )

                await asyncio.sleep(0)

                continue


            # =================================================
            # Hazard
            # =================================================

            hazard = final_result


            print(
                "[AI] result:",
                hazard
            )


            # =================================================
            # BLE 비동기 전송
            #
            # ACK를 기다리는 동안에도
            # 다음 AI 프레임 처리는 계속 가능
            # =================================================

            current_time = now_ms()


            if (
                current_time
                - last_ble_time
                >= BLE_INTERVAL_MS
            ):

                # 이전 BLE 작업이 끝났을 때만
                # 새로운 BLE 작업 시작

                if (
                    ble_task is None
                    or ble_task.done()
                ):

                    ble_task = asyncio.create_task(
                        send_hazard_ble(
                            sender,
                            hazard,
                            ai_result_time
                        )
                    )


                    last_ble_time = (
                        current_time
                    )


            # =================================================
            # 주기적 평균 AI FPS 출력
            # =================================================

            if (
                processed_frames > 0
                and processed_frames % 20 == 0
            ):

                avg_ai_time = (
                    total_ai_time
                    / processed_frames
                )


                avg_ai_fps = (
                    processed_frames
                    / total_ai_time
                    if total_ai_time > 0
                    else 0
                )


                elapsed_total = (
                    time.monotonic()
                    - stats_start_time
                )


                print()
                print(
                    "========== AI STATUS =========="
                )

                print(
                    f"Processed frames : "
                    f"{processed_frames}"
                )

                print(
                    f"Average AI time  : "
                    f"{avg_ai_time * 1000:.1f} ms/frame"
                )

                print(
                    f"Average AI FPS   : "
                    f"{avg_ai_fps:.2f}"
                )

                print(
                    f"Running time     : "
                    f"{elapsed_total:.1f} s"
                )

                print(
                    "==============================="
                )

                print()


            # event loop 제어권 반환
            await asyncio.sleep(0)


    except asyncio.CancelledError:

        print(
            "[SYSTEM] Task cancelled"
        )

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
        # 프로그램 종료
        # =================================================

        print(
            "[SYSTEM] 종료 처리 시작"
        )


        # ---------------------------------------------
        # 카메라 캡처 스레드 중지 요청
        # ---------------------------------------------

        capture_stop.set()


        # ---------------------------------------------
        # 카메라 종료
        # ---------------------------------------------

        if camera is not None:

            try:

                camera.stop()

                print(
                    "[SYSTEM] Camera stopped"
                )


            except Exception as e:

                print(
                    f"[SYSTEM] Camera 종료 오류: "
                    f"{type(e).__name__}: {repr(e)}"
                )


        # ---------------------------------------------
        # 캡처 스레드 종료 대기
        # ---------------------------------------------

        if (
            capture_thread is not None
            and capture_thread.is_alive()
        ):

            capture_thread.join(
                timeout=2.0
            )


        # ---------------------------------------------
        # 남은 BLE task 처리
        # ---------------------------------------------

        if (
            ble_task is not None
            and not ble_task.done()
        ):

            try:

                await ble_task

            except Exception:

                pass


        # ---------------------------------------------
        # BLE 연결 종료
        # ---------------------------------------------

        try:

            await sender.disconnect()


        except Exception as e:

            print(
                f"[SYSTEM] BLE 종료 오류: "
                f"{type(e).__name__}: {repr(e)}"
            )


        print(
            "[SYSTEM] 종료 완료"
        )


# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )


    except KeyboardInterrupt:

        print()

        print(
            "[SYSTEM] 사용자 종료"
        )