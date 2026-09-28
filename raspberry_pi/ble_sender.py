import asyncio
from datetime import datetime

from bleak import BleakClient, BleakScanner

from config import (
    DEVICE_ADDRESS,
    WRITE_CHARACTERISTIC_UUID,
    NOTIFY_CHARACTERISTIC_UUID,
    RETRY_DELAY
)


# =========================================================
# 설정
# =========================================================

SCAN_TIMEOUT = 3.0
DEVICE_SETTLE_DELAY = 0.5
CONNECT_TIMEOUT = 6.0
NOTIFY_TIMEOUT = 5.0


def log(message):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {message}")


class BLESender:
    def __init__(self):
        self.device_address = DEVICE_ADDRESS
        self.write_uuid = WRITE_CHARACTERISTIC_UUID
        self.notify_uuid = NOTIFY_CHARACTERISTIC_UUID

        self.client = None
        self.ack_event = asyncio.Event()

        # 최초 연결 로그를 한 번만 출력하기 위한 플래그
        self.first_connection = True


    # =====================================================
    # 연결 끊김 callback
    # =====================================================

    def _disconnected_callback(self, client):
        log("[BLE] 연결 끊김")


    # =====================================================
    # ESP32 검색
    # =====================================================

    async def find_device(self):
        try:
            device = await BleakScanner.find_device_by_address(
                self.device_address,
                timeout=SCAN_TIMEOUT
            )

        except Exception as e:
            log(
                f"[BLE] ERROR - ESP32 검색 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )

            return None


        if device is None:
            log(
                f"[BLE] ERROR - ESP32를 찾지 못했습니다: "
                f"{self.device_address}"
            )

            return None


        return device


    # =====================================================
    # ESP32 연결
    # =====================================================

    async def connect(self):

        # 최초 연결 시에만 출력
        if self.first_connection:
            log("[BLE] ESP32 연결 시작")


        # 이전 client가 남아 있으면 정리
        if self.client is not None:
            await self._safe_disconnect()


        device = await self.find_device()

        if device is None:
            return False


        # 장치 발견 직후 잠시 대기
        await asyncio.sleep(DEVICE_SETTLE_DELAY)


        # 매 연결 시도마다 새로운 BleakClient 생성
        self.client = BleakClient(
            device,
            disconnected_callback=self._disconnected_callback
        )


        try:
            # =================================================
            # 1. GATT 연결
            # =================================================

            await asyncio.wait_for(
                self.client.connect(),
                timeout=CONNECT_TIMEOUT
            )


            if not self.client.is_connected:
                log(
                    "[BLE] ERROR - "
                    "connect() 후 연결 상태가 False입니다."
                )

                await self._safe_disconnect()

                return False


            # =================================================
            # 2. GATT 서비스
            # =================================================

            services = self.client.services


            # =================================================
            # 3. Notify characteristic 확인
            # =================================================

            notify_char = services.get_characteristic(
                self.notify_uuid
            )

            if notify_char is None:
                log(
                    "[BLE] ERROR - "
                    "Notify characteristic을 찾지 못했습니다."
                )

                await self._safe_disconnect()

                return False


            # =================================================
            # 4. Write characteristic 확인
            # =================================================

            write_char = services.get_characteristic(
                self.write_uuid
            )

            if write_char is None:
                log(
                    "[BLE] ERROR - "
                    "Write characteristic을 찾지 못했습니다."
                )

                await self._safe_disconnect()

                return False


            # =================================================
            # 5. Notify 등록
            # =================================================

            await asyncio.wait_for(
                self.client.start_notify(
                    self.notify_uuid,
                    self._notification_handler
                ),
                timeout=NOTIFY_TIMEOUT
            )


            # 최초 연결 성공 로그는 딱 한 번만 출력
            if self.first_connection:
                log("[BLE] ESP32 연결 성공")
                self.first_connection = False


            return True


        # =====================================================
        # 연결 Timeout
        # =====================================================

        except asyncio.TimeoutError:
            log(
                f"[BLE] ERROR - 연결 TIMEOUT "
                f"(최대 {CONNECT_TIMEOUT:.1f}s)"
            )

            await self._safe_disconnect()

            return False


        # =====================================================
        # 기타 연결 오류
        # =====================================================

        except Exception as e:
            log(
                f"[BLE] ERROR - 연결 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )

            await self._safe_disconnect()

            return False


    # =====================================================
    # 실패한 client 정리
    # =====================================================

    async def _safe_disconnect(self):
        client = self.client

        if client is None:
            return


        try:
            if client.is_connected:
                try:
                    await client.disconnect()

                except Exception as e:
                    log(
                        f"[BLE] ERROR - disconnect 실패: "
                        f"{type(e).__name__}: {repr(e)}"
                    )


        except Exception as e:
            log(
                f"[BLE] ERROR - client 상태 확인 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )


        finally:
            self.client = None


    # =====================================================
    # 연결 재시도
    # =====================================================

    async def connect_with_retry(self):

        while True:
            success = await self.connect()

            if success:
                return

            # 실패 로그는 connect() 내부에서 이미 출력됨
            await asyncio.sleep(RETRY_DELAY)


    # =====================================================
    # ESP32 Notify callback
    # =====================================================

    def _notification_handler(self, sender, data):

        try:
            message = data.decode(
                "utf-8"
            ).strip()

        except Exception as e:
            log(
                f"[BLE] ERROR - Notify decode 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )

            return


        # ACK 정상 수신은 로그 출력 안 함
        if message == "ACK":
            self.ack_event.set()


    # =====================================================
    # BLE 패킷 전송
    # =====================================================

    async def send(self, packet):

        if not self.write_uuid:
            raise ValueError(
                "WRITE_CHARACTERISTIC_UUID가 "
                "설정되지 않았습니다."
            )


        # =================================================
        # 연결이 끊어진 경우 자동 재연결
        # =================================================

        if (
            self.client is None
            or not self.client.is_connected
        ):
            await self.connect_with_retry()


        try:
            await self.client.write_gatt_char(
                self.write_uuid,
                packet.encode("utf-8"),
                response=True
            )

            # 정상 전송 로그 없음
            return True


        except Exception as e:
            log(
                f"[BLE] ERROR - 전송 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )

            # 다음 전송 시 새 연결을 사용하도록 정리
            await self._safe_disconnect()

            return False


    # =====================================================
    # ACK 초기화
    # =====================================================

    def clear_ack(self):
        self.ack_event.clear()


    # =====================================================
    # ACK 대기
    # =====================================================

    async def wait_for_ack(
        self,
        timeout=1.0
    ):

        try:
            await asyncio.wait_for(
                self.ack_event.wait(),
                timeout=timeout
            )

            # 정상 ACK 로그 없음
            return True


        except asyncio.TimeoutError:
            log(
                f"[BLE] ERROR - ACK TIMEOUT "
                f"({timeout}s)"
            )

            return False


    # =====================================================
    # 프로그램 종료
    # =====================================================

    async def disconnect(self):

        if self.client is None:
            return


        client = self.client


        try:
            if client.is_connected:

                # Notify 종료
                try:
                    await client.stop_notify(
                        self.notify_uuid
                    )

                except Exception as e:
                    log(
                        f"[BLE] ERROR - stop_notify 실패: "
                        f"{type(e).__name__}: {repr(e)}"
                    )


                # GATT 연결 종료
                try:
                    await client.disconnect()

                except Exception as e:
                    log(
                        f"[BLE] ERROR - 연결 종료 실패: "
                        f"{type(e).__name__}: {repr(e)}"
                    )


        finally:
            self.client = None