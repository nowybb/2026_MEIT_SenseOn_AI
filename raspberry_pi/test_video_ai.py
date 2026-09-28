import asyncio
import threading
import sys
import time
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

# 실제 영상 파일명으로 수정
VIDEO_PATH = ROOT_DIR / "test_videos" / "Test_DANGER.mp4"

# 성능 측정할 때는 False 추천
LOOP_VIDEO = False

# ESP32 없을 때 BLE 연결 시도 시간
BLE_CONNECT_TIMEOUT = 10.0


# =========================================================
# 메인
# =========================================================

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
                "[ERROR] 영상 파일을 열 수 없습니다:"
            )

            print(VIDEO_PATH)

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


        print()
        print("========== VIDEO INFO ==========")
        print(f"Source FPS     : {source_fps:.2f}")
        print(f"Total Frames   : {total_source_frames}")
        print(f"Duration       : {original_duration:.2f} sec")
        print("================================")


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
                "[SYSTEM] AI + 영상 모드로 계속 실행"
            )

            print(
                f"[BLE] {e}"
            )


        # =================================================
        # 6. 영상 반복
        # =================================================

        loop_count = 1

        while True:

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
                f"========== LOOP {loop_count} START =========="
            )


            # =================================================
            # 실시간 영상 처리
            # =================================================

            while True:

                # 실제 경과 시간
                real_elapsed = (
                    time.perf_counter()
                    - video_start_real_time
                )


                # 현재 시점에 해당하는 원본 프레임
                target_frame_index = int(
                    real_elapsed
                    * source_fps
                )


                # 영상 끝
                if (
                    target_frame_index
                    >= total_source_frames
                ):
                    break


                # =================================================
                # 뒤처졌으면 중간 프레임 DROP
                # =================================================

                if (
                    target_frame_index
                    > current_frame_index
                ):

                    frames_to_drop = (
                        target_frame_index
                        - current_frame_index
                    )

                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        target_frame_index
                    )

                    dropped_frames += (
                        frames_to_drop
                    )

                    current_frame_index = (
                        target_frame_index
                    )


                # =================================================
                # 현재 프레임 읽기
                # =================================================

                ret, frame = cap.read()

                if not ret:
                    break


                current_frame_index += 1
                processed_frames += 1


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
                    ai_end - ai_start
                )


                total_ai_time += (
                    ai_time
                )


                ai_result_time = (
                    now_ms()
                )


                # =================================================
                # 스트리밍
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

                if (
                    state == "READY"
                    and final_result is not None
                ):

                    hazard = (
                        final_result
                    )


                    print(
                        "[AI]",
                        hazard
                    )


                    # =================================================
                    # BLE 연결된 경우
                    # =================================================

                    if ble_enabled:

                        try:

                            packet = (
                                encode_hazard(
                                    hazard
                                )
                            )


                            sender.clear_ack()


                            send_success = (
                                await sender.send(
                                    packet
                                )
                            )


                            if send_success:

                                ack_received = (
                                    await sender.wait_for_ack()
                                )


                                if ack_received:

                                    ack_time = (
                                        now_ms()
                                    )


                                    e2e_latency = (
                                        calc_latency_ms(
                                            ai_result_time,
                                            ack_time
                                        )
                                    )


                                    print(
                                        f"[LATENCY] "
                                        f"{e2e_latency:.2f} ms"
                                    )


                                    # 기존 senseon_log.csv 저장
                                    save_log(
                                        hazard,
                                        e2e_latency_ms=
                                        e2e_latency
                                    )


                                else:

                                    print(
                                        "[BLE] ACK 수신 실패"
                                    )


                            else:

                                print(
                                    "[BLE] 전송 실패"
                                )


                        except Exception as e:

                            print(
                                f"[BLE ERROR] {e}"
                            )


            # =================================================
            # 영상 1회 종료 후 최종 성능 계산
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

                avg_ai_fps = (
                    1.0
                    / avg_ai_time
                )

            else:

                avg_ai_time = 0.0
                avg_ai_fps = 0.0


            # =================================================
            # 최종 결과 Remote Shell 출력
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