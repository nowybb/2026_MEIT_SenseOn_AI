import asyncio
import threading
import sys
import time
import csv
from pathlib import Path

import cv2


# =========================================================
# 프로젝트 경로
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
RASPBERRY_PI_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(RASPBERRY_PI_DIR))


from stream_server import update_frame, run_stream_server
from protocol import encode_hazard
from ble_sender import BLESender
from latency import now_ms, calc_latency_ms
from logger import save_log
from senseon_pipeline import FrameAnalyzer


# =========================================================
# 설정
# =========================================================

MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"

VIDEO_PATH = (
    ROOT_DIR
    / "test_videos"
    / "Test_real_640.mp4"
)

# 성능 측정할 때는 False 추천
LOOP_VIDEO = False

# ESP32 없을 때 연결 시도 시간
BLE_CONNECT_TIMEOUT = 10.0

# main.py와 동일하게 BLE 전송은 최대 5Hz
# ACK 대기 때문에 영상/AI 루프가 멈추지 않도록 비동기 task로 전송
BLE_INTERVAL_MS = 200


# =========================================================
# 테스트 CSV 로그
# =========================================================

TEST_LOG_PATH = (
    RASPBERRY_PI_DIR
    / "test_video_log.csv"
)


TEST_LOG_HEADER = [
    "video_name",
    "loop",
    "event",
    "video_time_sec",
    "state",
    "object",
    "direction",
    "risk",
    "ttc",
    "latency_ms",
    "processed_frames",
    "dropped_frames",
    "processing_fps",
    "avg_ai_time_ms",
    "avg_ai_fps",
    "frame_drop_rate_percent"
]


def append_test_log(
    video_name,
    loop_count,
    event,
    video_time_sec="",
    state="",
    object_name="",
    direction="",
    risk="",
    ttc="",
    latency_ms="",
    processed_frames="",
    dropped_frames="",
    processing_fps="",
    avg_ai_time_ms="",
    avg_ai_fps="",
    frame_drop_rate_percent=""
):

    # 파일이 없거나 비어 있으면 헤더 작성
    need_header = (
        not TEST_LOG_PATH.exists()
        or TEST_LOG_PATH.stat().st_size == 0
    )

    with open(
        TEST_LOG_PATH,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        if need_header:
            writer.writerow(
                TEST_LOG_HEADER
            )

        writer.writerow([
            video_name,
            loop_count,
            event,
            video_time_sec,
            state,
            object_name,
            direction,
            risk,
            ttc,
            latency_ms,
            processed_frames,
            dropped_frames,
            processing_fps,
            avg_ai_time_ms,
            avg_ai_fps,
            frame_drop_rate_percent
        ])


# =========================================================
# Hazard 값 가져오기
# =========================================================

def get_hazard_value(
    hazard,
    key,
    default=""
):

    if hazard is None:
        return default

    # dictionary 형태
    if isinstance(
        hazard,
        dict
    ):

        value = hazard.get(
            key,
            default
        )

    # 객체 형태
    else:

        value = getattr(
            hazard,
            key,
            default
        )

    if value is None:
        return ""

    return value


# =========================================================
# BLE 비동기 전송
#
# main.py와 동일한 방식:
# ACK를 기다리는 동안에도 영상/AI 처리는 계속 진행
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

        print(
            f"[BLE] packet: {packet}"
        )

        sender.clear_ack()

        send_success = await sender.send(
            packet
        )

        if not send_success:

            print(
                "[BLE] 전송 실패"
            )

            return

        # SAFE 패킷은 send()가 끝나는 순간 ESP32로 전달됨.
        # ACK는 latency 측정을 위해 뒤에서 기다리되 AI 루프를 막지 않음.
        ack_received = await sender.wait_for_ack()

        if not ack_received:

            print(
                "[BLE] ACK 수신 실패"
            )

            return

        ack_time = now_ms()

        e2e_latency = calc_latency_ms(
            ai_result_time,
            ack_time
        )

        latency_state["last_e2e_latency"] = (
            e2e_latency
        )

        print(
            f"[LATENCY] {e2e_latency:.2f} ms"
        )

        save_log(
            hazard,
            e2e_latency_ms=e2e_latency
        )

    except asyncio.CancelledError:
        raise

    except Exception as e:

        print(
            f"[BLE ERROR] {e}"
        )


# =========================================================
# 메인
# =========================================================

async def main():

    sender = BLESender()

    cap = None
    ble_enabled = False

    # main.py와 동일한 BLE 비동기 전송 상태
    ble_task = None
    last_ble_time = 0

    latency_state = {
        "last_e2e_latency": None
    }

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
        # 2. 영상 파일 열기
        # =================================================

        cap = cv2.VideoCapture(
            str(VIDEO_PATH)
        )

        if not cap.isOpened():

            print(
                "[ERROR] 영상 파일을 열 수 없습니다:"
            )

            print(
                VIDEO_PATH
            )

            return


        # =================================================
        # 3. 영상 정보
        # =================================================

        source_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if source_fps <= 0:

            source_fps = 30.0

        total_source_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        original_duration = (
            total_source_frames
            / source_fps
        )

        video_name = (
            VIDEO_PATH.name
        )

        print()

        print(
            "========== VIDEO INFO =========="
        )

        print(
            f"Source FPS     : "
            f"{source_fps:.2f}"
        )

        print(
            f"Total Frames   : "
            f"{total_source_frames}"
        )

        print(
            f"Duration       : "
            f"{original_duration:.2f} sec"
        )

        print(
            "================================"
        )

        print(
            f"[LOG] CSV 저장 위치: "
            f"{TEST_LOG_PATH}"
        )


        # =================================================
        # 4. 스트리밍 서버 시작
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
        # 5. BLE 연결 시도
        # =================================================

        print(
            "[SYSTEM] ESP32 BLE 연결 시도"
        )

        try:

            await asyncio.wait_for(
                sender.connect_with_retry(),
                timeout=BLE_CONNECT_TIMEOUT
            )

            ble_enabled = True

            print(
                "[SYSTEM] ESP32 BLE 연결 완료"
            )

        except Exception as e:

            ble_enabled = False

            print(
                "[SYSTEM] ESP32 연결 실패"
            )

            print(
                "[SYSTEM] AI + 영상 + 스트리밍 모드로 계속 실행"
            )

            print(
                f"[BLE] {e}"
            )


        # =================================================
        # 6. 영상 반복
        # =================================================

        loop_count = 1

        while True:

            # 영상 처음으로 이동
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            video_start_real_time = (
                time.perf_counter()
            )

            processed_frames = 0
            dropped_frames = 0

            total_ai_time = 0.0

            current_frame_index = 0


            print()

            print(
                f"========== LOOP "
                f"{loop_count} START =========="
            )


            # =================================================
            # CSV : 테스트 시작 표시
            # =================================================

            append_test_log(
                video_name=video_name,
                loop_count=loop_count,
                event="START",
                video_time_sec=0.0
            )


            # =================================================
            # 실시간 영상 처리
            # =================================================

            while True:

                # -------------------------------------------------
                # 실제 경과 시간
                # -------------------------------------------------

                real_elapsed = (
                    time.perf_counter()
                    - video_start_real_time
                )


                # -------------------------------------------------
                # 현재 시점에 해당하는 원본 프레임
                # -------------------------------------------------

                target_frame_index = int(
                    real_elapsed
                    * source_fps
                )


                # -------------------------------------------------
                # 영상 끝
                # -------------------------------------------------

                if (
                    target_frame_index
                    >= total_source_frames
                ):

                    break


                # =================================================
                # 뒤처진 프레임 DROP
                #
                # seek(cap.set) 하지 않고
                # grab()으로 빠르게 넘김
                # =================================================

                while (
                    current_frame_index
                    < target_frame_index
                ):

                    grabbed = cap.grab()

                    if not grabbed:
                        break

                    current_frame_index += 1
                    dropped_frames += 1


                # =================================================
                # 현재 프레임 읽기
                # =================================================

                ret, frame = cap.read()

                if not ret:
                    break

                current_frame_index += 1
                processed_frames += 1


                # =================================================
                # 영상 자체의 현재 시점
                # =================================================

                current_video_time = (
                    current_frame_index
                    / source_fps
                )


                # =================================================
                # AI 처리
                # =================================================

                ai_start = (
                    time.perf_counter()
                )

                final_result, state, annotated_frame = (
                    await asyncio.to_thread(
                        analyzer.process,
                        frame,
                        current_video_time,
                        annotate=True
                    )
                )

                ai_end = (
                    time.perf_counter()
                )

                ai_time = (
                    ai_end
                    - ai_start
                )

                total_ai_time += (
                    ai_time
                )

                ai_result_time = (
                    now_ms()
                )


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
                # AI 결과
                # =================================================

                if state != "READY":

                    print(
                        f"[AI] state={state}"
                    )

                    await asyncio.sleep(0)
                    continue


                # main.py와 동일하게 READY이면
                # final_result가 None이어도 그대로 hazard로 사용
                # protocol.py에서 None은 none,CENTER,SAFE,None 으로 변환됨
                hazard = final_result

                print(
                    "[AI]",
                    hazard
                )


                # ---------------------------------------------
                # Hazard 정보 (CSV용)
                # ---------------------------------------------

                object_name = get_hazard_value(
                    hazard,
                    "object"
                )

                direction = get_hazard_value(
                    hazard,
                    "direction"
                )

                risk = get_hazard_value(
                    hazard,
                    "risk"
                )

                ttc = get_hazard_value(
                    hazard,
                    "ttc"
                )

                # None은 실제 BLE 패킷상 SAFE 해제 상태
                if hazard is None:
                    object_name = "none"
                    direction = "CENTER"
                    risk = "SAFE"
                    ttc = "None"


                # =================================================
                # BLE 비동기 전송
                #
                # 중요:
                # 여기서 ACK를 직접 await하지 않음.
                # main.py처럼 별도 task가 ACK를 기다리므로
                # 테스트 영상의 다음 프레임/SAFE 프레임을 놓치지 않음.
                # =================================================

                current_time = now_ms()

                # SAFE(None 포함)는 모터 OFF 명령이므로 최우선 전송
                is_safe = (
                    hazard is None
                    or str(risk).upper() == "SAFE"
                )

                if ble_enabled and is_safe:

                    # 이전 CAUTION/DANGER ACK 대기 task가 남아 있으면
                    # SAFE가 밀리지 않도록 취소 후 즉시 전송
                    if (
                        ble_task is not None
                        and not ble_task.done()
                    ):

                        ble_task.cancel()

                        try:
                            await ble_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass

                    print(
                        "[BLE] SAFE 우선 전송 -> 모터 OFF"
                    )

                    ble_task = asyncio.create_task(
                        send_hazard_ble(
                            sender,
                            hazard,
                            ai_result_time,
                            {"enabled": ble_enabled},
                            latency_state
                        )
                    )

                    last_ble_time = current_time

                elif (
                    ble_enabled
                    and current_time - last_ble_time >= BLE_INTERVAL_MS
                ):

                    if (
                        ble_task is None
                        or ble_task.done()
                    ):

                        ble_task = asyncio.create_task(
                            send_hazard_ble(
                                sender,
                                hazard,
                                ai_result_time,
                                {"enabled": ble_enabled},
                                latency_state
                            )
                        )

                        last_ble_time = current_time


                # =================================================
                # CSV : AI 결과 저장
                # =================================================

                append_test_log(
                    video_name=video_name,
                    loop_count=loop_count,
                    event="AI_RESULT",

                    video_time_sec=round(
                        current_video_time,
                        3
                    ),

                    state=state,

                    object_name=object_name,

                    direction=direction,

                    risk=risk,

                    ttc=ttc,

                    latency_ms=(
                        round(
                            latency_state["last_e2e_latency"],
                            2
                        )
                        if latency_state["last_e2e_latency"] is not None
                        else ""
                    )
                )

                # BLE task가 실행될 기회를 줌
                await asyncio.sleep(0)


            # =================================================
            # 영상 1회 종료 후 성능 계산
            # =================================================

            actual_duration = (
                time.perf_counter()
                - video_start_real_time
            )

            total_seen_frames = (
                processed_frames
                + dropped_frames
            )

            if total_seen_frames > 0:

                drop_rate = (
                    dropped_frames
                    / total_seen_frames
                    * 100.0
                )

            else:

                drop_rate = 0.0


            if actual_duration > 0:

                processing_fps = (
                    processed_frames
                    / actual_duration
                )

            else:

                processing_fps = 0.0


            if processed_frames > 0:

                avg_ai_time = (
                    total_ai_time
                    / processed_frames
                )

                if avg_ai_time > 0:

                    avg_ai_fps = (
                        1.0
                        / avg_ai_time
                    )

                else:

                    avg_ai_fps = 0.0

            else:

                avg_ai_time = 0.0
                avg_ai_fps = 0.0


            # =================================================
            # CSV : 테스트 종료 표시 + 최종 성능
            # =================================================

            append_test_log(
                video_name=video_name,
                loop_count=loop_count,
                event="END",

                video_time_sec=round(
                    original_duration,
                    3
                ),

                processed_frames=
                processed_frames,

                dropped_frames=
                dropped_frames,

                processing_fps=round(
                    processing_fps,
                    3
                ),

                avg_ai_time_ms=round(
                    avg_ai_time * 1000.0,
                    3
                ),

                avg_ai_fps=round(
                    avg_ai_fps,
                    3
                ),

                frame_drop_rate_percent=round(
                    drop_rate,
                    3
                )
            )


            # =================================================
            # 최종 결과 출력
            # =================================================

            print()
            print()

            print(
                "========== FINAL RESULT =========="
            )

            print(
                f"Original Duration : "
                f"{original_duration:.2f} sec"
            )

            print(
                f"Actual Duration   : "
                f"{actual_duration:.2f} sec"
            )

            print(
                f"Source FPS        : "
                f"{source_fps:.2f}"
            )

            print(
                f"Processing FPS    : "
                f"{processing_fps:.2f}"
            )

            print(
                f"Average AI Time   : "
                f"{avg_ai_time * 1000:.1f} ms/frame"
            )

            print(
                f"Average AI FPS    : "
                f"{avg_ai_fps:.2f}"
            )

            print(
                f"Processed Frames  : "
                f"{processed_frames}"
            )

            print(
                f"Dropped Frames    : "
                f"{dropped_frames}"
            )

            print(
                f"Frame Drop Rate   : "
                f"{drop_rate:.2f}%"
            )

            print(
                "=================================="
            )

            print(
                f"[LOG] 테스트 결과 저장 완료: "
                f"{TEST_LOG_PATH}"
            )

            print()
            print()


            # =================================================
            # 반복 여부
            # =================================================

            if not LOOP_VIDEO:

                break

            loop_count += 1

            print(
                "[SYSTEM] 영상 처음부터 다시 재생"
            )


    except KeyboardInterrupt:

        print()

        print(
            "[SYSTEM] 사용자 종료"
        )


    finally:

        # 남아 있는 BLE 전송 task가 있으면 마무리
        if (
            ble_task is not None
            and not ble_task.done()
        ):

            try:
                await ble_task
            except Exception:
                pass


        if cap is not None:

            cap.release()


        if ble_enabled:

            try:

                await sender.disconnect()

            except Exception:

                pass


        print(
            "[SYSTEM] 종료 완료"
        )


# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )