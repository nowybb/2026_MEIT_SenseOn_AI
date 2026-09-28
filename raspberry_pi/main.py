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


MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n_toy_finetuned.pt"


# =========================================================
# 스트리밍 설정
# =========================================================

# True  : 브라우저 스트리밍 사용
# False : 스트리밍 끄기
STREAM_ENABLED = True


# =========================================================
# BLE 설정
# =========================================================

# 최대 BLE 전송 주기
# 200ms = 최대 5Hz
BLE_INTERVAL_MS = 200

# BLE 연결 1회 시도 최대 시간
BLE_CONNECT_TIMEOUT = 10.0

# 연결 실패 후 재시도 간격
BLE_RETRY_INTERVAL = 3.0


# =========================================================
# BLE 재연결 루프
#
# ESP32가 없어도 AI / 카메라 / 스트리밍은 계속 실행
# BLE는 백그라운드에서 계속 연결 재시도
# =========================================================

async def ble_reconnect_loop(
    sender,
    ble_state
):

    while True:

        # 이미 연결되어 있으면 대기
        if ble_state["enabled"]:

            await asyncio.sleep(
                BLE_RETRY_INTERVAL
            )

            continue


        try:

            print(
                "[BLE] ESP32 연결 시도"
            )


            # 이전 연결 상태 정리
            try:

                await sender.disconnect()

            except Exception:

                pass


            await asyncio.wait_for(
                sender.connect_with_retry(),
                timeout=BLE_CONNECT_TIMEOUT
            )


            ble_state["enabled"] = True


            print(
                "[BLE] ESP32 연결 완료"
            )


        except asyncio.CancelledError:

            print(
                "[BLE] 재연결 task 종료"
            )

            raise


        except Exception as e:

            ble_state["enabled"] = False


            print(
                f"[BLE] 연결 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )


            print(
                f"[BLE] "
                f"{BLE_RETRY_INTERVAL:.0f}초 후 재시도"
            )


        await asyncio.sleep(
            BLE_RETRY_INTERVAL
        )


# =========================================================
# BLE 비동기 전송
# =========================================================

async def send_hazard_ble(
    sender,
    hazard,
    ai_result_time,
    ble_state,
    latency_state
):

    try:

        packet = encode_hazard(
            hazard
        )


        sender.clear_ack()


        send_success = await sender.send(
            packet
        )


        if not send_success:

            print(
                "[BLE] 전송 실패"
            )

            ble_state["enabled"] = False

            return


        ack_received = (
            await sender.wait_for_ack()
        )


        if not ack_received:

            print(
                "[BLE] ACK 수신 실패"
            )

            ble_state["enabled"] = False

            return


        # ---------------------------------------------
        # ACK 수신 시점
        # ---------------------------------------------

        ack_time = now_ms()


        # ---------------------------------------------
        # End-to-End Latency
        #
        # AI 판단 완료
        # -> BLE
        # -> ESP32 처리
        # -> ACK 수신
        # ---------------------------------------------

        e2e_latency = calc_latency_ms(
            ai_result_time,
            ack_time
        )


        # 가장 최근 latency 저장
        latency_state["last_e2e_latency"] = (
            e2e_latency
        )


        print(
            f"[LATENCY] End-to-End: "
            f"{e2e_latency:.1f} ms"
        )


        # ---------------------------------------------
        # 기존 CSV 로그 저장
        # ---------------------------------------------

        save_log(
            hazard,
            e2e_latency_ms=e2e_latency
        )


    except asyncio.CancelledError:

        raise


    except Exception as e:

        print(
            f"[BLE ERROR] "
            f"{type(e).__name__}: {repr(e)}"
        )


        ble_state["enabled"] = False


# =========================================================
# MAIN
# =========================================================

async def main():

    sender = BLESender()


    # =====================================================
    # BLE 상태
    # =====================================================

    ble_state = {
        "enabled": False
    }


    # =====================================================
    # E2E Latency 상태
    # =====================================================

    latency_state = {
        "last_e2e_latency": None
    }


    camera = None

    capture_thread = None
    capture_stop = threading.Event()

    ble_task = None
    ble_reconnect_task = None

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


                timestamp = (
                    now_ms()
                    / 1000.0
                )


                with frame_lock:

                    latest_frame = frame

                    latest_timestamp = (
                        timestamp
                    )

                    latest_frame_id += 1


            except Exception as e:

                if not capture_stop.is_set():

                    print(
                        f"[CAMERA ERROR] "
                        f"{type(e).__name__}: "
                        f"{repr(e)}"
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

        if STREAM_ENABLED:

            stream_thread = threading.Thread(
                target=run_stream_server,
                daemon=True
            )


            stream_thread.start()


            print(
                "[SYSTEM] Live stream server started"
            )


        else:

            print(
                "[SYSTEM] Live stream disabled"
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


            await asyncio.sleep(
                0.01
            )


        if STREAM_ENABLED:

            update_frame(
                initial_frame
            )


        print(
            "[SYSTEM] 초기 카메라 화면 준비 완료"
        )


        # =================================================
        # 6. BLE 백그라운드 재연결 시작
        #
        # ESP32가 없어도 AI는 계속 실행
        # =================================================

        print(
            "[SYSTEM] BLE 백그라운드 연결 시작"
        )


        ble_reconnect_task = (
            asyncio.create_task(
                ble_reconnect_loop(
                    sender,
                    ble_state
                )
            )
        )


        # =================================================
        # 7. 통계값
        # =================================================

        processed_frames = 0

        last_processed_frame_id = 0

        total_ai_time = 0.0

        total_dropped_frames = 0

        stats_start_time = (
            time.monotonic()
        )


        # =================================================
        # 8. AI 통합 루프
        #
        # 항상 가장 최신 프레임만 분석
        # =================================================

        while True:

            # ---------------------------------------------
            # 최신 프레임 가져오기
            # ---------------------------------------------

            with frame_lock:

                current_frame_id = (
                    latest_frame_id
                )


                if (
                    latest_frame is None
                    or current_frame_id
                    == last_processed_frame_id
                ):

                    frame = None


                else:

                    frame = (
                        latest_frame.copy()
                    )


                    timestamp = (
                        latest_timestamp
                    )


            if frame is None:

                await asyncio.sleep(
                    0.001
                )

                continue


            # =================================================
            # Frame Drop 계산
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

                total_dropped_frames += (
                    dropped_frames
                )


                print(
                    f"[FRAME] "
                    f"Dropped "
                    f"{dropped_frames} frame(s)"
                )


            last_processed_frame_id = (
                current_frame_id
            )


            # =================================================
            # AI 추론
            # =================================================

            ai_start = (
                time.perf_counter()
            )


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


            total_ai_time += (
                ai_elapsed
            )


            processed_frames += 1


            # AI 판단 완료 시점
            ai_result_time = (
                now_ms()
            )


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
            #
            # STREAM_ENABLED=True일 때만 실행
            # =================================================

            if STREAM_ENABLED:

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


                await asyncio.sleep(
                    0
                )

                continue


            # =================================================
            # Hazard
            # =================================================

            hazard = (
                final_result
            )


            print(
                "[AI] result:",
                hazard
            )


            # =================================================
            # BLE 비동기 전송
            #
            # BLE가 연결된 경우만 실행
            # =================================================

            current_time = (
                now_ms()
            )


            if (
                ble_state["enabled"]
                and current_time
                - last_ble_time
                >= BLE_INTERVAL_MS
            ):

                if (
                    ble_task is None
                    or ble_task.done()
                ):

                    ble_task = (
                        asyncio.create_task(
                            send_hazard_ble(
                                sender,
                                hazard,
                                ai_result_time,
                                ble_state,
                                latency_state
                            )
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


                total_seen_frames = (
                    processed_frames
                    + total_dropped_frames
                )


                if total_seen_frames > 0:

                    drop_rate = (
                        total_dropped_frames
                        / total_seen_frames
                        * 100.0
                    )


                else:

                    drop_rate = 0.0


                print()

                print(
                    "========== AI STATUS =========="
                )


                print(
                    f"Processed frames : "
                    f"{processed_frames}"
                )


                print(
                    f"Dropped frames   : "
                    f"{total_dropped_frames}"
                )


                print(
                    f"Frame drop rate  : "
                    f"{drop_rate:.2f}%"
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
                    f"Stream enabled   : "
                    f"{STREAM_ENABLED}"
                )


                print(
                    f"BLE connected    : "
                    f"{ble_state['enabled']}"
                )


                # -----------------------------------------
                # 최근 E2E latency 출력
                # -----------------------------------------

                if (
                    latency_state[
                        "last_e2e_latency"
                    ]
                    is not None
                ):

                    print(
                        f"Last E2E latency : "
                        f"{latency_state['last_e2e_latency']:.1f} ms"
                    )


                else:

                    print(
                        "Last E2E latency : "
                        "N/A"
                    )


                print(
                    "==============================="
                )


                print()


            await asyncio.sleep(
                0
            )


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
        # 카메라 캡처 스레드 중지
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
                    f"{type(e).__name__}: "
                    f"{repr(e)}"
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
        # BLE 전송 task 종료
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
        # BLE 재연결 task 종료
        # ---------------------------------------------

        if (
            ble_reconnect_task is not None
            and not ble_reconnect_task.done()
        ):

            ble_reconnect_task.cancel()


            try:

                await ble_reconnect_task


            except asyncio.CancelledError:

                pass


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
                f"{type(e).__name__}: "
                f"{repr(e)}"
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