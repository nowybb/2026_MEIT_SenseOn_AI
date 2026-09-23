import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime

from config import (
    DEVICE_NAME,
    WRITE_CHARACTERISTIC_UUID,
    NOTIFY_CHARACTERISTIC_UUID,
    RETRY_DELAY
)


def log(message):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {message}")


class BLESender:
    def __init__(self):
        self.device_name = DEVICE_NAME
        self.write_uuid = WRITE_CHARACTERISTIC_UUID
        self.notify_uuid = NOTIFY_CHARACTERISTIC_UUID

        self.client = None

        # ACK 도착 여부 확인
        self.ack_event = asyncio.Event()


    async def find_device(self):
        log(f"[BLE] {self.device_name} 검색 중...")

        devices = await BleakScanner.discover(timeout=10.0)

        for device in devices:
            if device.name == self.device_name:
                log(
                    f"[BLE] 장치 발견: "
                    f"{device.name} / {device.address}"
                )
                return device

        log("[BLE] 장치를 찾지 못했습니다.")
        return None


    async def connect(self):
        device = await self.find_device()

        if device is None:
            return False

        self.client = BleakClient(device)

        try:
            # 1. GATT 연결
            log("[BLE] GATT 연결 시도...")

            await asyncio.wait_for(
                self.client.connect(),
                timeout=10.0
            )

            log(
                f"[BLE] GATT 연결 완료: "
                f"{self.client.is_connected}"
            )

            # 2. Notify 등록
            log(
                f"[BLE] Notify 등록 시도: "
                f"{self.notify_uuid}"
            )

            await asyncio.wait_for(
                self.client.start_notify(
                    self.notify_uuid,
                    self._notification_handler
                ),
                timeout=10.0
            )

            log("[BLE] Notify 등록 완료")
            log("[BLE] 연결 성공")

            return True

        except asyncio.TimeoutError:
            log("[BLE] 연결 과정 TIMEOUT")

            if (
                self.client is not None
                and self.client.is_connected
            ):
                try:
                    await self.client.disconnect()
                except Exception:
                    pass

            return False

        except Exception as e:
            log(
                f"[BLE] 연결 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )

            if (
                self.client is not None
                and self.client.is_connected
            ):
                try:
                    await self.client.disconnect()
                except Exception:
                    pass

            return False


    async def connect_with_retry(self):
        while True:
            if await self.connect():
                return

            log(
                f"[BLE] {RETRY_DELAY}초 후 재연결"
            )

            await asyncio.sleep(RETRY_DELAY)


    def _notification_handler(self, sender, data):
        try:
            message = data.decode("utf-8").strip()
        except Exception as e:
            log(
                f"[BLE] Notify decode 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )
            return

        log(f"[BLE] 수신: {message}")

        if message == "ACK":
            self.ack_event.set()


    async def send(self, packet):
        if not self.write_uuid:
            raise ValueError(
                "WRITE_CHARACTERISTIC_UUID가 "
                "설정되지 않았습니다."
            )

        if (
            self.client is None
            or not self.client.is_connected
        ):
            log("[BLE] 연결 없음. 재연결 시도")
            await self.connect_with_retry()

        try:
            await self.client.write_gatt_char(
                self.write_uuid,
                packet.encode("utf-8")
            )

            log(f"[BLE] 전송 성공: {packet}")
            return True

        except Exception as e:
            log(
                f"[BLE] 전송 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )

            return False


    def clear_ack(self):
        # 새 패킷 전송 전 이전 ACK 상태 초기화
        self.ack_event.clear()


    async def wait_for_ack(self, timeout=1.0):
        try:
            await asyncio.wait_for(
                self.ack_event.wait(),
                timeout=timeout
            )

            log("[BLE] ACK 수신 성공")
            return True

        except asyncio.TimeoutError:
            log("[BLE] ACK 수신 실패")
            return False


    async def disconnect(self):
        if (
            self.client is not None
            and self.client.is_connected
        ):
            try:
                await self.client.stop_notify(
                    self.notify_uuid
                )
            except Exception:
                pass

            try:
                await self.client.disconnect()
            except Exception as e:
                log(
                    f"[BLE] 연결 종료 중 오류: "
                    f"{type(e).__name__}: {repr(e)}"
                )
                return

            log("[BLE] 연결 종료")