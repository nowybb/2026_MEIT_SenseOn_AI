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
MODEL_PATH = ROOT / "ai1" / "yolo11n.pt"


class CameraReader:
    def __init__(self, backend, source, width, height):
        self.backend = backend
        self.camera = None
        self.cv2 = None
        if backend == "picamera2":
            from picamera2 import Picamera2
            self.camera = Picamera2()
            cfg = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                buffer_count=3,
            )
            self.camera.configure(cfg)
            self.camera.start()
            # Picamera2 RGB888 배열은 OpenCV에서 사용할 수 있는 BGR 순서.
            # 반드시 실제 기기에서 색상을 확인할 것.
        else:
            import cv2
            self.cv2 = cv2
            if not (source.isdigit() or source.startswith("/dev/video")):
                raise ValueError("실시간 입력은 카메라 번호 또는 /dev/video* 만 허용합니다")
            device = int(source) if source.isdigit() else source
            self.camera = cv2.VideoCapture(device)
            if not self.camera.isOpened():
                self.camera.release()
                raise RuntimeError(f"카메라를 열지 못했습니다: {source}")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지원하지 않는 드라이버에선 무시됨

    def read(self):
        if self.backend == "picamera2":
            return True, self.camera.capture_array("main")
        return self.camera.read()

    def close(self):
        if self.camera is not None:
            if self.backend == "picamera2":
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()


async def ble_test(address):
    """AI/카메라와 관계없이 전자과 ESP32 수신·진동만 확인."""
    ble = ESP32BLE(address=address)
    try:
        await ble.connect()
        for packet in (
            "car,LEFT,DANGER,1.8",
            "car,CENTER,CAUTION,3.0",
            "car,RIGHT,DANGER,1.8",
            RELEASE_PACKET,
            "motorcycle,CENTER,CAUTION,None",  # 긴 패킷 테스트
            RELEASE_PACKET,
        ):
            print("[BLE TEST]", packet, len(packet.encode("utf-8")), "bytes")
            await ble.send(packet)
            await asyncio.sleep(0.8)
    finally:
        await ble.close()


async def run_live(args):
    import cv2
    from senseon_pipeline import FrameAnalyzer

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO 가중치가 없습니다: {MODEL_PATH}")
    camera = None
    writer = None
    ble = ESP32BLE(address=args.ble_address) if not args.no_ble else None
    successful = False
    stopped_normally = False

    try:
        # 장치 연결 테스트를 먼저 시행: 실패 시 YOLO/카메라 시작 안 함.
        if ble is not None:
            await ble.connect()
        analyzer = FrameAnalyzer(MODEL_PATH, mirror_direction=args.mirror_direction)
        camera = CameraReader(args.backend, args.source, args.width, args.height)
        start = time.monotonic()
        last_send = -1e9
        previous_state_key = None
        frame_count = 0

        print("[LIVE] 실제 카메라 입력 시작 (MP4 입력 아님). Ctrl+C로 종료")
        while True:
            ok, frame = await asyncio.to_thread(camera.read)
            if not ok or frame is None:
                raise RuntimeError("카메라 프레임을 읽지 못했습니다. 장애를 SAFE로 취급하지 않습니다")
            capture_at = time.monotonic()
            # CPU 추론 중에도 이벤트 루프가 BLE 콜백을 처리할 수 있게 한다.
            result, state, annotated = await asyncio.to_thread(
                analyzer.process, frame, capture_at - start,
                args.show or bool(args.record),
            )
            finished_at = time.monotonic()
            frame_count += 1
            latency = finished_at - capture_at
            if latency > 0.8:
                print(f"[LATENCY WARNING] 처리 {latency:.2f}초. ESP32 1초 타임아웃 주의")

            if state == "UNKNOWN":
                # 객체는 보이지만 추적/이력이 불충분함: SAFE 전송 금지.
                # ESP32는 유효 패킷이 끊기면 1초 후 진동 OFF하도록 구성되어 있음.
                if frame_count % 15 == 0:
                    print("[AI] 분석 미완료: SAFE라고 단정하지 않아 새 명령 보류")
            else:
                packet = make_ble_packet(result)
                now = time.monotonic()
                state_key = ("none", "CENTER", "SAFE") if result is None else (
                    result["object"], result["direction"], result["risk"]
                )
                # TTC 숫자만 매 프레임 달라졌다고 매번 전송하지 않음.
                if (state_key != previous_state_key or now - last_send >= args.send_interval):
                    if ble is not None:
                        # ACK 오류/연결 오류 발생 시 시스템 장애로 중단: 자동 복구 주장하지 않음.
                        await ble.send(packet)
                    else:
                        print("[DRY RUN]", packet)
                    previous_state_key = state_key
                    last_send = time.monotonic()

            if args.record:
                if writer is None:
                    h, w = annotated.shape[:2]
                    path = Path(args.record)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                        max(1.0, args.record_fps), (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"시연 영상 저장을 열지 못했습니다: {path}")
                writer.write(annotated)

            if args.show:
                cv2.imshow("SenseOn LIVE - Debug", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_normally = True
                    break
            if frame_count % 30 == 0:
                print(f"[LIVE] {frame_count} frames | average {frame_count/(time.monotonic()-start):.1f} FPS "
                      f"| latest AI latency {latency:.2f}s")
            if args.frames and frame_count >= args.frames:
                stopped_normally = True
                break
        successful = True
    finally:
        # 정상 종료 시 즉시 SAFE 전송. 카메라/AI/통신 오류면 정상이라고 주장하지 않음.
        if ble is not None:
            if stopped_normally and successful:
                try:
                    await ble.send(RELEASE_PACKET)
                except Exception as exc:
                    print(f"[STOP] 해제 패킷 전송 실패: {exc}")
            await ble.close()  # 연결 끊기면 ESP32 코드가 모터 OFF 처리
        if writer is not None:
            writer.release()
        if camera is not None:
            camera.close()
        if args.show:
            cv2.destroyAllWindows()
        print("[STOP] 카메라/BLE 종료. 장애 발생 시 시스템 상태를 반드시 확인하세요")


def parse_args():
    parser = argparse.ArgumentParser(description="SenseOn Raspberry Pi real-time demo")
    parser.add_argument("--ble-test", action="store_true", help="고정 패킷만 전송; 카메라/YOLO 사용 안 함")
    parser.add_argument("--no-ble", action="store_true", help="카메라 AI만 로컬 점검; BLE는 출력")
    parser.add_argument("--ble-address", default=None, help="같은 이름의 장치가 여러 개면 ESP32 BLE 주소 지정")
    parser.add_argument("--backend", choices=("opencv", "picamera2"), default="opencv")
    parser.add_argument("--source", default="0", help="OpenCV USB 카메라 번호 또는 /dev/videoN")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mirror-direction", action="store_true", help="LEFT/RIGHT 전송만 반전 (실물 기준 확인 후)")
    parser.add_argument("--show", action="store_true", help="디버깅 영상 창 표시; GUI 없는 Pi에서는 사용 X")
    parser.add_argument("--record", default=None, help="시연 화면 녹화 경로, 예: demo.mp4")
    parser.add_argument("--record-fps", type=float, default=10.0, help="녹화 영상 목표 FPS; 실제 추론 FPS 확인 필요")
    parser.add_argument("--send-interval", type=float, default=0.35, help="동일 상태 재전송 최소 간격")
    parser.add_argument("--frames", type=int, default=0, help="테스트용 처리 프레임 제한, 0=무제한")
    return parser.parse_args()


async def main():
    args = parse_args()
    if args.send_interval <= 0 or args.send_interval >= 0.8:
        raise ValueError("--send-interval은 0~0.8초 미만으로 설정하세요 (ESP32 타임아웃 1초)")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("카메라 해상도가 잘못되었습니다")
    if args.ble_test:
        await ble_test(args.ble_address)
    else:
        await run_live(args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[STOP] 사용자 종료. ESP32 연결 해제/타임아웃을 확인하세요")
