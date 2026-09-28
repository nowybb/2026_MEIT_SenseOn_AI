"""실제 카메라 -> YOLO/BoT-SORT -> AI2 -> Raspberry Pi BLE -> ESP32.

프로토타입. 실제 도로에서 안전 장치로 의존하지 말 것.
"""

import argparse
import asyncio
from pathlib import Path
import time

from senseon_protocol import RELEASE_PACKET, make_ble_packet
from pi_ble import ESP32BLE


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ai1" / "yolo11n_toy_finetuned.pt"


# ============================================================
# Camera
# ============================================================

class CameraReader:
    def __init__(self, backend, source, width, height):
        self.backend = backend
        self.camera = None
        self.cv2 = None

        if backend == "picamera2":
            from picamera2 import Picamera2

            self.camera = Picamera2()

            cfg = self.camera.create_video_configuration(
                main={
                    "size": (width, height),
                    "format": "RGB888",
                },
                buffer_count=3,
            )

            self.camera.configure(cfg)
            self.camera.start()

        else:
            import cv2

            self.cv2 = cv2

            if not (
                source.isdigit()
                or source.startswith("/dev/video")
            ):
                raise ValueError(
                    "실시간 입력은 카메라 번호 또는 "
                    "/dev/video* 만 허용합니다"
                )

            device = (
                int(source)
                if source.isdigit()
                else source
            )

            self.camera = cv2.VideoCapture(device)

            if not self.camera.isOpened():
                self.camera.release()
                raise RuntimeError(
                    f"카메라를 열지 못했습니다: {source}"
                )

            self.camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                width,
            )

            self.camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                height,
            )

            # 가능한 경우 오래된 frame이 버퍼에 쌓이는 것을 방지
            self.camera.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

    def read(self):
        if self.backend == "picamera2":
            return (
                True,
                self.camera.capture_array("main"),
            )

        return self.camera.read()

    def close(self):
        if self.camera is None:
            return

        if self.backend == "picamera2":
            self.camera.stop()
            self.camera.close()

        else:
            self.camera.release()


# ============================================================
# BLE 단독 테스트
# ============================================================

async def ble_test(address):
    """AI/카메라 없이 ESP32 수신·진동만 확인."""

    ble = ESP32BLE(address=address)

    try:
        await ble.connect()

        for packet in (
            "car,LEFT,DANGER,1.8",
            "car,CENTER,CAUTION,3.0",
            "car,RIGHT,DANGER,1.8",
            RELEASE_PACKET,
            "motorcycle,CENTER,CAUTION,None",
            RELEASE_PACKET,
        ):
            print(
                "[BLE TEST]",
                packet,
                len(packet.encode("utf-8")),
                "bytes",
            )

            await ble.send(packet)
            await asyncio.sleep(0.8)

    finally:
        await ble.close()


# ============================================================
# 화면 표시용 간단한 상태 표시
# ============================================================

def draw_live_status(
    frame,
    result,
    state,
    ai_latency,
    ai_fps,
):
    import cv2

    out = frame.copy()

    # --------------------------------------------------------
    # AI 결과
    # --------------------------------------------------------

    if result is not None:
        status_text = (
            f"HAZARD: "
            f"{result['object']} "
            f"{result['direction']} "
            f"{result['risk']}"
        )

    elif state == "UNKNOWN":
        status_text = "ANALYSIS PENDING"

    else:
        status_text = "HAZARD: NONE"

    # --------------------------------------------------------
    # 상단 배경
    # --------------------------------------------------------

    cv2.rectangle(
        out,
        (0, 0),
        (out.shape[1], 65),
        (30, 30, 30),
        -1,
    )

    cv2.putText(
        out,
        status_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        out,
        (
            f"AI {ai_fps:.1f} FPS "
            f"| {ai_latency * 1000:.0f} ms"
        ),
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return out


# ============================================================
# 실제 실시간 실행
# ============================================================

async def run_live(args):
    import cv2
    from senseon_pipeline import FrameAnalyzer

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO 가중치가 없습니다: {MODEL_PATH}"
        )

    camera = None
    writer = None

    ble = (
        ESP32BLE(address=args.ble_address)
        if not args.no_ble
        else None
    )

    # --------------------------------------------------------
    # 공유 상태
    #
    # frame을 queue에 여러 장 쌓지 않고
    # 항상 최신 frame 하나만 유지
    # --------------------------------------------------------

    latest_frame = {
        "frame": None,
        "timestamp": 0.0,
        "seq": 0,
    }

    latest_ai = {
        "result": None,
        "state": "UNKNOWN",
        "latency": 0.0,
        "fps": 0.0,
    }

    stop_event = asyncio.Event()

    stopped_normally = False

    capture_task = None
    ai_task = None

    try:

        # ====================================================
        # BLE 연결
        # ====================================================

        if ble is not None:
            await ble.connect()

        # ====================================================
        # AI / Camera 초기화
        # ====================================================

        analyzer = FrameAnalyzer(
            MODEL_PATH,
            mirror_direction=args.mirror_direction,
        )

        camera = CameraReader(
            args.backend,
            args.source,
            args.width,
            args.height,
        )

        start = time.monotonic()

        print(
            "[LIVE] 카메라 / AI 비동기 처리 시작 "
            "(Ctrl+C 또는 q로 종료)"
        )

        # ====================================================
        # 1. Camera task
        #
        # AI와 관계없이 계속 최신 frame을 읽는다.
        # ====================================================

        async def capture_loop():

            while not stop_event.is_set():

                ok, frame = await asyncio.to_thread(
                    camera.read
                )

                if not ok or frame is None:
                    raise RuntimeError(
                        "카메라 프레임을 읽지 못했습니다"
                    )

                # 오래된 프레임은 저장하지 않고
                # 최신 프레임으로 덮어씀
                latest_frame["frame"] = frame
                latest_frame["timestamp"] = (
                    time.monotonic()
                )
                latest_frame["seq"] += 1

                # 다른 coroutine에 실행 기회 부여
                await asyncio.sleep(0)

        # ====================================================
        # 2. AI task
        #
        # 최신 frame만 가져와 YOLO + BoT-SORT + AI2 수행
        # ====================================================

        async def ai_loop():

            nonlocal writer

            last_processed_seq = -1

            last_send = -1e9
            previous_state_key = None

            ai_frame_count = 0
            ai_start = time.monotonic()

            while not stop_event.is_set():

                current_seq = latest_frame["seq"]

                # 아직 새 frame이 없음
                if (
                    current_seq == 0
                    or current_seq == last_processed_seq
                ):
                    await asyncio.sleep(0.001)
                    continue

                # ------------------------------------------------
                # 가장 최신 frame 하나만 가져옴
                # ------------------------------------------------

                frame = latest_frame["frame"].copy()
                capture_at = latest_frame["timestamp"]

                last_processed_seq = current_seq

                process_start = time.monotonic()

                # ------------------------------------------------
                # YOLO + BoT-SORT + AI2
                #
                # 화면 자체는 raw frame으로 출력할 것이므로
                # show 때문에 annotate=True로 만들 필요 없음.
                #
                # 녹화가 필요할 때만 annotated frame 생성.
                # ------------------------------------------------

                result, state, annotated = (
                    await asyncio.to_thread(
                        analyzer.process,
                        frame,
                        capture_at - start,
                        bool(args.record),
                    )
                )

                process_end = time.monotonic()

                latency = (
                    process_end
                    - process_start
                )

                ai_frame_count += 1

                elapsed = (
                    process_end
                    - ai_start
                )

                ai_fps = (
                    ai_frame_count / elapsed
                    if elapsed > 0
                    else 0.0
                )

                # ------------------------------------------------
                # 최신 AI 결과 업데이트
                # ------------------------------------------------

                latest_ai["result"] = result
                latest_ai["state"] = state
                latest_ai["latency"] = latency
                latest_ai["fps"] = ai_fps

                # ------------------------------------------------
                # latency warning
                # ------------------------------------------------

                if latency > 0.8:
                    print(
                        f"[LATENCY WARNING] "
                        f"AI 처리 {latency:.2f}초. "
                        f"ESP32 1초 타임아웃 주의"
                    )

                # ------------------------------------------------
                # BLE
                # ------------------------------------------------

                if state == "UNKNOWN":

                    if ai_frame_count % 15 == 0:
                        print(
                            "[AI] 분석 미완료: "
                            "새 명령 보류"
                        )

                else:

                    packet = make_ble_packet(
                        result
                    )

                    now = time.monotonic()

                    state_key = (
                        ("none", "CENTER", "SAFE")
                        if result is None
                        else (
                            result["object"],
                            result["direction"],
                            result["risk"],
                        )
                    )

                    if (
                        state_key
                        != previous_state_key
                        or now - last_send
                        >= args.send_interval
                    ):

                        if ble is not None:

                            await ble.send(
                                packet
                            )

                        else:

                            print(
                                "[DRY RUN]",
                                packet,
                            )

                        previous_state_key = (
                            state_key
                        )

                        last_send = (
                            time.monotonic()
                        )

                # ------------------------------------------------
                # 영상 녹화
                #
                # 녹화는 AI 처리 frame 기준.
                # ------------------------------------------------

                if args.record:

                    if writer is None:

                        h, w = annotated.shape[:2]

                        path = Path(
                            args.record
                        )

                        path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        writer = cv2.VideoWriter(
                            str(path),
                            cv2.VideoWriter_fourcc(
                                *"mp4v"
                            ),
                            max(
                                1.0,
                                args.record_fps,
                            ),
                            (w, h),
                        )

                        if not writer.isOpened():
                            raise RuntimeError(
                                "시연 영상 저장을 "
                                "열지 못했습니다: "
                                f"{path}"
                            )

                    writer.write(
                        annotated
                    )

                # ------------------------------------------------
                # 로그
                # ------------------------------------------------

                if ai_frame_count % 30 == 0:

                    print(
                        f"[AI] "
                        f"{ai_frame_count} frames | "
                        f"{ai_fps:.1f} FPS | "
                        f"{latency * 1000:.0f} ms"
                    )

                # ------------------------------------------------
                # frame 제한
                # ------------------------------------------------

                if (
                    args.frames
                    and ai_frame_count
                    >= args.frames
                ):
                    stop_event.set()
                    return

        # ====================================================
        # Task 시작
        # ====================================================

        capture_task = asyncio.create_task(
            capture_loop()
        )

        ai_task = asyncio.create_task(
            ai_loop()
        )

        # ====================================================
        # 3. 화면 표시 loop
        #
        # AI 처리 완료를 기다리지 않는다.
        # 최신 카메라 frame을 계속 표시한다.
        # ====================================================

        while not stop_event.is_set():

            # Capture / AI 내부 오류 확인
            for task in (
                capture_task,
                ai_task,
            ):
                if task.done():

                    exc = task.exception()

                    if exc is not None:
                        raise exc

            if args.show:

                frame = latest_frame["frame"]

                if frame is not None:

                    display = draw_live_status(
                        frame,
                        latest_ai["result"],
                        latest_ai["state"],
                        latest_ai["latency"],
                        latest_ai["fps"],
                    )

                    cv2.imshow(
                        "SenseOn LIVE",
                        display,
                    )

                    if (
                        cv2.waitKey(1)
                        & 0xFF
                        == ord("q")
                    ):
                        stopped_normally = True
                        stop_event.set()
                        break

                # 약 30 FPS 화면 출력
                await asyncio.sleep(
                    1.0 / args.display_fps
                )

            else:

                # GUI 없는 Raspberry Pi 실행
                await asyncio.sleep(
                    0.02
                )

        if args.frames:
            stopped_normally = True

    finally:

        stop_event.set()

        # ----------------------------------------------------
        # task 종료
        # ----------------------------------------------------

        tasks = [
            task
            for task in (
                capture_task,
                ai_task,
            )
            if task is not None
        ]

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        # ----------------------------------------------------
        # 정상 종료 시 SAFE
        # ----------------------------------------------------

        if ble is not None:

            if stopped_normally:

                try:
                    await ble.send(
                        RELEASE_PACKET
                    )

                except Exception as exc:
                    print(
                        "[STOP] 해제 패킷 "
                        f"전송 실패: {exc}"
                    )

            await ble.close()

        # ----------------------------------------------------
        # 자원 해제
        # ----------------------------------------------------

        if writer is not None:
            writer.release()

        if camera is not None:
            camera.close()

        if args.show:
            cv2.destroyAllWindows()

        print(
            "[STOP] 카메라/BLE 종료. "
            "장애 발생 시 시스템 상태를 확인하세요"
        )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SenseOn Raspberry Pi real-time demo"
        )
    )

    parser.add_argument(
        "--ble-test",
        action="store_true",
        help="고정 패킷만 전송; 카메라/YOLO 사용 안 함",
    )

    parser.add_argument(
        "--no-ble",
        action="store_true",
        help="카메라 AI만 로컬 점검; BLE는 출력",
    )

    parser.add_argument(
        "--ble-address",
        default=None,
        help="같은 이름의 장치가 여러 개면 ESP32 BLE 주소 지정",
    )

    parser.add_argument(
        "--backend",
        choices=(
            "opencv",
            "picamera2",
        ),
        default="opencv",
    )

    parser.add_argument(
        "--source",
        default="0",
        help="OpenCV USB 카메라 번호 또는 /dev/videoN",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--mirror-direction",
        action="store_true",
        help="LEFT/RIGHT 전송만 반전",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="실시간 카메라 화면 표시",
    )

    parser.add_argument(
        "--record",
        default=None,
        help="시연 영상 녹화 경로",
    )

    parser.add_argument(
        "--record-fps",
        type=float,
        default=10.0,
    )

    # 새로 추가
    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="화면 표시 목표 FPS",
    )

    parser.add_argument(
        "--send-interval",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=0,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

async def main():

    args = parse_args()

    if (
        args.send_interval <= 0
        or args.send_interval >= 0.8
    ):
        raise ValueError(
            "--send-interval은 "
            "0~0.8초 미만으로 설정하세요"
        )

    if (
        args.width <= 0
        or args.height <= 0
    ):
        raise ValueError(
            "카메라 해상도가 잘못되었습니다"
        )

    if args.display_fps <= 0:
        raise ValueError(
            "--display-fps는 0보다 커야 합니다"
        )

    if args.ble_test:
        await ble_test(
            args.ble_address
        )

    else:
        await run_live(
            args
        )


if __name__ == "__main__":

    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        print(
            "[STOP] 사용자 종료. "
            "ESP32 연결 해제/타임아웃을 확인하세요"
        )