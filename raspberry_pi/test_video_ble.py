import asyncio
import threading
import time
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


# =========================================================
# 설정
# =========================================================

MODEL_PATH = ROOT_DIR / "ai1" / "yolo11n.pt"
VIDEO_PATH = ROOT_DIR / "test_videos" / "Test_DANGER_640.mp4"

# BLE 최대 전송 주기
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

        ack_time = now_ms()

        e2e_latency = calc_latency_ms(
            ai_result_time,
            ack_time
        )

        print(
            f"[LATENCY] "
            f"{e2e_latency:.1f} ms"
        )

        save_log(
            hazard,
            e2e_latency_ms=e2e_latency
        )

    except Exception as e:
        print(
            "[BLE ERROR]",
            e
        )


# =========================================================
# MAIN
# =========================================================

async def main():

    sender = BLESender()

    cap = None
    ble_task = None

    last_ble_time = 0

    try:

        # =================================================
        # 1. AI 모델
        # =================================================

        print("[SYSTEM] AI 모델 로드")

        analyzer = FrameAnalyzer(
            str(MODEL_PATH)
        )

        print("[SYSTEM] AI 모델 준비 완료")


        # =================================================
        # 2. 영상 열기
        # =================================================

        cap = cv2.VideoCapture(
            str(VIDEO_PATH)
        )

        if not cap.isOpened():

            print(
                "[ERROR] 영상 파일을 열 수 없음:",
                VIDEO_PATH
            )

            return


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:
            fps = 30.0


        duration = (
            frame_count / fps
            if frame_count > 0
            else 0
        )


        print(
            f"[VIDEO] FPS = {fps:.2f}"
        )

        print(
            f"[VIDEO] Frames = {frame_count}"
        )

        print(
            f"[VIDEO] Duration = {duration:.2f}s"
        )


        # =================================================
        # 3. Flask 스트리밍
        # =================================================

        stream_thread = threading.Thread(
            target=run_stream_server,
            daemon=True
        )

        stream_thread.start()

        print(
            "[SYSTEM] Stream server started"
        )


        # =================================================
        # 4. BLE 연결
        # =================================================

        print(
            "[SYSTEM] ESP32 BLE 연결"
        )

        await sender.connect_with_retry()

        print(
            "[SYSTEM] ESP32 BLE 연결 완료"
        )


        # =================================================
        # 5. 재생 기준 시각
        # =================================================

        playback_start = time.monotonic()

        processed_frames = 0
        skipped_frames = 0


        # =================================================
        # 6. 영상 루프
        # =================================================

        while True:

            # ---------------------------------------------
            # 현재 실제 경과시간
            # ---------------------------------------------

            elapsed = (
                time.monotonic()
                - playback_start
            )


            # ---------------------------------------------
            # 현재 시간에 해당하는 목표 frame 계산
            #
            # 예:
            # 2초 경과 + 30FPS
            # -> frame 60이 최신 프레임
            # ---------------------------------------------

            target_frame = int(
                elapsed * fps
            )


            # 영상 끝
            if target_frame >= frame_count:

                print(
                    "[SYSTEM] 영상 재생 완료"
                )

                break


            # ---------------------------------------------
            # 현재 VideoCapture 위치
            # ---------------------------------------------

            current_frame = int(
                cap.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            )


            # ---------------------------------------------
            # AI가 뒤처졌다면 프레임 skip
            # ---------------------------------------------

            if target_frame > current_frame:

                skipped = (
                    target_frame
                    - current_frame
                )

                skipped_frames += skipped

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    target_frame
                )


            # ---------------------------------------------
            # 최신 프레임 읽기
            # ---------------------------------------------

            ret, frame = cap.read()

            if not ret:
                break


            processed_frames += 1


            # ---------------------------------------------
            # 영상 기준 timestamp
            # ---------------------------------------------

            timestamp = (
                target_frame / fps
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
            # 스트리밍 업데이트
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

                await asyncio.sleep(0)

                continue


            hazard = final_result


            print(
                "[AI]",
                hazard
            )


            # =================================================
            # BLE 전송
            #
            # 영상 처리는 멈추지 않음
            # =================================================

            current_time = now_ms()


            if (
                current_time
                - last_ble_time
                >= BLE_INTERVAL_MS
            ):

                # 이전 BLE 작업이 끝났을 때만
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


            # event loop에 제어권 반환
            await asyncio.sleep(0)


        # =================================================
        # 남은 BLE task 종료 대기
        # =================================================

        if (
            ble_task is not None
            and not ble_task.done()
        ):

            await ble_task


        # =================================================
        # 결과
        # =================================================

        total_time = (
            time.monotonic()
            - playback_start
        )


        print()
        print(
            "================================"
        )

        print(
            f"[RESULT] 영상 길이: "
            f"{duration:.2f}s"
        )

        print(
            f"[RESULT] 실제 재생시간: "
            f"{total_time:.2f}s"
        )

        print(
            f"[RESULT] AI 처리 프레임: "
            f"{processed_frames}"
        )

        print(
            f"[RESULT] Skip 프레임: "
            f"{skipped_frames}"
        )

        if total_time > 0:

            processing_fps = (
                processed_frames
                / total_time
            )

            print(
                f"[RESULT] AI 처리 FPS: "
                f"{processing_fps:.2f}"
            )

        print(
            "================================"
        )


    except KeyboardInterrupt:

        print()
        print(
            "[SYSTEM] 사용자 종료"
        )


    finally:

        if cap is not None:
            cap.release()

        await sender.disconnect()

        print(
            "[SYSTEM] 종료 완료"
        )


if __name__ == "__main__":

    asyncio.run(
        main()
    )