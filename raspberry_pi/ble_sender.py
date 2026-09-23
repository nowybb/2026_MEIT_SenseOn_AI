import asyncio
from bleak import BleakClient, BleakScanner
from datetime import datetime
import time

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
        self.ack_event = asyncio.Event()


    def _disconnected_callback(self, client):
        log("[BLE] *** 연결 끊김 callback 발생 ***")


    async def find_device(self):
        log(f"[BLE] {self.device_name} 검색 시작")

        start = time.perf_counter()

        try:
            devices = await BleakScanner.discover(timeout=10.0)

        except Exception as e:
            log(
                f"[BLE] 스캔 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )
            return None

        elapsed = time.perf_counter() - start

        log(
            f"[BLE] 스캔 완료 "
            f"({elapsed:.3f}s, 발견 {len(devices)}개)"
        )

        for device in devices:
            if device.name == self.device_name:
                log(
                    f"[BLE] 대상 장치 발견"
                )
                log(
                    f"[BLE] name    : {device.name}"
                )
                log(
                    f"[BLE] address : {device.address}"
                )
                return device

        log("[BLE] 대상 장치를 찾지 못했습니다.")
        return None


    async def connect(self):
        log("[BLE] ===== 연결 절차 시작 =====")

        device = await self.find_device()

        if device is None:
            log("[BLE] 연결 중단: 장치 검색 실패")
            return False

        self.client = BleakClient(
            device,
            disconnected_callback=self._disconnected_callback
        )

        try:
            # -------------------------
            # 1. GATT 연결 시작
            # -------------------------
            log("[BLE] GATT connect() 호출")

            connect_start = time.perf_counter()

            await asyncio.wait_for(
                self.client.connect(),
                timeout=15.0
            )

            connect_elapsed = (
                time.perf_counter() - connect_start
            )

            log(
                f"[BLE] connect() 반환 "
                f"({connect_elapsed:.3f}s)"
            )

            log(
                f"[BLE] is_connected = "
                f"{self.client.is_connected}"
            )

            if not self.client.is_connected:
                log(
                    "[BLE] connect()는 반환됐지만 "
                    "is_connected=False"
                )
                return False


            # -------------------------
            # 2. 서비스 접근 확인
            # -------------------------
            log("[BLE] GATT 서비스 확인 시작")

            service_start = time.perf_counter()

            services = self.client.services

            service_elapsed = (
                time.perf_counter() - service_start
            )

            log(
                f"[BLE] 서비스 접근 완료 "
                f"({service_elapsed:.3f}s)"
            )

            log(
                f"[BLE] Service 개수: "
                f"{len(services.services)}"
            )

            # 서비스 / characteristic 목록 출력
            for service in services:
                log(
                    f"[BLE] SERVICE: {service.uuid}"
                )

                for char in service.characteristics:
                    log(
                        f"[BLE]   CHAR: {char.uuid} "
                        f"properties={char.properties}"
                    )


            # -------------------------
            # 3. Notify characteristic 확인
            # -------------------------
            log(
                f"[BLE] Notify UUID 검색: "
                f"{self.notify_uuid}"
            )

            notify_char = services.get_characteristic(
                self.notify_uuid
            )

            if notify_char is None:
                log(
                    "[BLE] ERROR: Notify characteristic "
                    "찾지 못함"
                )
                return False

            log(
                f"[BLE] Notify characteristic 발견: "
                f"{notify_char.uuid}"
            )

            log(
                f"[BLE] Notify properties: "
                f"{notify_char.properties}"
            )


            # -------------------------
            # 4. Write characteristic 확인
            # -------------------------
            log(
                f"[BLE] Write UUID 검색: "
                f"{self.write_uuid}"
            )

            write_char = services.get_characteristic(
                self.write_uuid
            )

            if write_char is None:
                log(
                    "[BLE] ERROR: Write characteristic "
                    "찾지 못함"
                )
                return False

            log(
                f"[BLE] Write characteristic 발견: "
                f"{write_char.uuid}"
            )

            log(
                f"[BLE] Write properties: "
                f"{write_char.properties}"
            )


            # -------------------------
            # 5. Notify 등록
            # -------------------------
            log("[BLE] start_notify() 호출")

            notify_start = time.perf_counter()

            await asyncio.wait_for(
                self.client.start_notify(
                    self.notify_uuid,
                    self._notification_handler
                ),
                timeout=10.0
            )

            notify_elapsed = (
                time.perf_counter() - notify_start
            )

            log(
                f"[BLE] Notify 등록 완료 "
                f"({notify_elapsed:.3f}s)"
            )

            log("[BLE] ===== 연결 성공 =====")

            return True


        except asyncio.TimeoutError:
            log("[BLE] TIMEOUT 발생")

            if (
                self.client is not None
                and self.client.is_connected
            ):
                log(
                    "[BLE] timeout 후 disconnect 시도"
                )

                try:
                    await self.client.disconnect()
                    log("[BLE] disconnect 완료")

                except Exception as e:
                    log(
                        f"[BLE] disconnect 실패: "
                        f"{type(e).__name__}: {repr(e)}"
                    )

            return False


        except Exception as e:
            log(
                f"[BLE] 예외 발생"
            )

            log(
                f"[BLE] exception type: "
                f"{type(e).__name__}"
            )

            log(
                f"[BLE] exception repr: "
                f"{repr(e)}"
            )

            if self.client is not None:
                log(
                    f"[BLE] 예외 시 is_connected: "
                    f"{self.client.is_connected}"
                )

            if (
                self.client is not None
                and self.client.is_connected
            ):
                log(
                    "[BLE] 예외 발생 후 disconnect 시도"
                )

                try:
                    await self.client.disconnect()
                    log("[BLE] disconnect 완료")

                except Exception as disconnect_error:
                    log(
                        f"[BLE] disconnect 실패: "
                        f"{type(disconnect_error).__name__}: "
                        f"{repr(disconnect_error)}"
                    )

            return False


    async def connect_with_retry(self):
        attempt = 1

        while True:
            log(
                f"[BLE] ===== 연결 시도 #{attempt} ====="
            )

            if await self.connect():
                return

            log(
                f"[BLE] 연결 시도 #{attempt} 실패"
            )

            log(
                f"[BLE] {RETRY_DELAY}초 후 재연결"
            )

            await asyncio.sleep(RETRY_DELAY)

            attempt += 1


    def _notification_handler(self, sender, data):
        try:
            message = data.decode("utf-8").strip()

        except Exception as e:
            log(
                f"[BLE] Notify decode 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )
            return

        log(
            f"[BLE] Notify 수신 "
            f"sender={sender}, data={data!r}"
        )

        log(
            f"[BLE] Notify message: {message}"
        )

        if message == "ACK":
            log("[BLE] ACK event set")
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
            log(
                "[BLE] send() 호출 시 연결 없음"
            )

            await self.connect_with_retry()

        log(
            f"[BLE] write 시작: {packet}"
        )

        log(
            f"[BLE] write UUID: "
            f"{self.write_uuid}"
        )

        try:
            start = time.perf_counter()

            await self.client.write_gatt_char(
                self.write_uuid,
                packet.encode("utf-8")
            )

            elapsed = time.perf_counter() - start

            log(
                f"[BLE] write 완료 "
                f"({elapsed:.3f}s)"
            )

            log(
                f"[BLE] 전송 성공: {packet}"
            )

            return True

        except Exception as e:
            log(
                f"[BLE] 전송 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )

            log(
                f"[BLE] 전송 실패 후 "
                f"is_connected="
                f"{self.client.is_connected}"
            )

            return False


    def clear_ack(self):
        self.ack_event.clear()
        log("[BLE] ACK event clear")


    async def wait_for_ack(self, timeout=1.0):
        log(
            f"[BLE] ACK 대기 시작 "
            f"(timeout={timeout}s)"
        )

        try:
            await asyncio.wait_for(
                self.ack_event.wait(),
                timeout=timeout
            )

            log("[BLE] ACK 수신 성공")
            return True

        except asyncio.TimeoutError:
            log("[BLE] ACK 대기 TIMEOUT")
            return False


    async def disconnect(self):
        if self.client is None:
            log(
                "[BLE] disconnect: client 없음"
            )
            return

        log(
            f"[BLE] disconnect 시작 "
            f"(is_connected={self.client.is_connected})"
        )

        if self.client.is_connected:

            try:
                log("[BLE] stop_notify 시작")

                await self.client.stop_notify(
                    self.notify_uuid
                )

                log("[BLE] stop_notify 완료")

            except Exception as e:
                log(
                    f"[BLE] stop_notify 실패: "
                    f"{type(e).__name__}: {repr(e)}"
                )

            try:
                log("[BLE] disconnect() 호출")

                await self.client.disconnect()

                log("[BLE] 연결 종료 완료")

            except Exception as e:
                log(
                    f"[BLE] 연결 종료 중 오류: "
                    f"{type(e).__name__}: {repr(e)}"
                )