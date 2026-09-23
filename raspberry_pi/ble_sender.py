import asyncio
import time
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

# 장치 발견 직후 BlueZ/BLE 상태가 정리될 시간을 조금 줌
DEVICE_SETTLE_DELAY = 0.5

# 정상 연결은 이전 테스트에서 약 2~3초였으므로
# 실패 시 너무 오래 기다리지 않도록 제한
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

        # ESP32 ACK 수신 여부
        self.ack_event = asyncio.Event()


    # =====================================================
    # 연결 끊김 callback
    # =====================================================

    def _disconnected_callback(self, client):
        log("[BLE] *** 연결 끊김 callback 발생 ***")


    # =====================================================
    # ESP32 검색
    # =====================================================

    async def find_device(self):
        log(f"[BLE] ESP32 검색 시작: {self.device_address}")

        start = time.perf_counter()

        try:
            device = await BleakScanner.find_device_by_address(
                self.device_address,
                timeout=3.0
            )

        except Exception as e:
            log(
                f"[BLE] 검색 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )
            return None

        elapsed = time.perf_counter() - start

        if device is None:
            log(
                f"[BLE] ESP32를 찾지 못했습니다. "
                f"({elapsed:.3f}s)"
            )
            return None

        log(
            f"[BLE] ESP32 발견 "
            f"({elapsed:.3f}s)"
        )
        log(f"[BLE] address : {device.address}")
        log(f"[BLE] name    : {device.name}")

        return device


    # =====================================================
    # ESP32 연결
    # =====================================================

    async def connect(self):
        log("[BLE] ===== 연결 절차 시작 =====")

        device = await self.find_device()

        if device is None:
            log("[BLE] 연결 중단: 장치 검색 실패")
            return False


        # -------------------------------------------------
        # 장치 발견 직후 0.5초 안정화
        # -------------------------------------------------

        log(
            f"[BLE] 장치 발견 후 "
            f"{DEVICE_SETTLE_DELAY:.1f}초 안정화 대기"
        )

        await asyncio.sleep(DEVICE_SETTLE_DELAY)


        # -------------------------------------------------
        # Bleak Client 생성
        # -------------------------------------------------

        self.client = BleakClient(
            device,
            disconnected_callback=self._disconnected_callback
        )

        try:
            # =============================================
            # 1. GATT 연결
            # =============================================

            log("[BLE] GATT connect() 호출")

            connect_start = time.perf_counter()

            await asyncio.wait_for(
                self.client.connect(),
                timeout=CONNECT_TIMEOUT
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
                    "[BLE] connect() 반환 후 "
                    "is_connected=False"
                )
                return False


            # =============================================
            # 2. GATT 서비스 확인
            # =============================================

            log("[BLE] GATT 서비스 확인 시작")

            services = self.client.services

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


            # =============================================
            # 3. Notify characteristic 확인
            # =============================================

            log(
                f"[BLE] Notify UUID 검색: "
                f"{self.notify_uuid}"
            )

            notify_char = services.get_characteristic(
                self.notify_uuid
            )

            if notify_char is None:
                log(
                    "[BLE] ERROR: "
                    "Notify characteristic 찾지 못함"
                )
                await self._safe_disconnect()
                return False

            log(
                f"[BLE] Notify characteristic 발견: "
                f"{notify_char.uuid}"
            )

            log(
                f"[BLE] Notify properties: "
                f"{notify_char.properties}"
            )


            # =============================================
            # 4. Write characteristic 확인
            # =============================================

            log(
                f"[BLE] Write UUID 검색: "
                f"{self.write_uuid}"
            )

            write_char = services.get_characteristic(
                self.write_uuid
            )

            if write_char is None:
                log(
                    "[BLE] ERROR: "
                    "Write characteristic 찾지 못함"
                )
                await self._safe_disconnect()
                return False

            log(
                f"[BLE] Write characteristic 발견: "
                f"{write_char.uuid}"
            )

            log(
                f"[BLE] Write properties: "
                f"{write_char.properties}"
            )


            # =============================================
            # 5. Notify 등록
            # =============================================

            log("[BLE] start_notify() 호출")

            notify_start = time.perf_counter()

            await asyncio.wait_for(
                self.client.start_notify(
                    self.notify_uuid,
                    self._notification_handler
                ),
                timeout=NOTIFY_TIMEOUT
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


        # =================================================
        # Timeout
        # =================================================

        except asyncio.TimeoutError:
            log(
                f"[BLE] 연결 과정 TIMEOUT "
                f"(connect 최대 {CONNECT_TIMEOUT:.1f}s)"
            )

            await self._safe_disconnect()

            return False


        # =================================================
        # 기타 예외
        # =================================================

        except Exception as e:
            log("[BLE] 연결 중 예외 발생")

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
                    f"[BLE] 예외 시 is_connected = "
                    f"{self.client.is_connected}"
                )

            await self._safe_disconnect()

            return False


    # =====================================================
    # 안전한 연결 해제
    # =====================================================

    async def _safe_disconnect(self):
        if self.client is None:
            return

        try:
            if self.client.is_connected:
                log("[BLE] 기존 연결 정리 시작")

                await self.client.disconnect()

                log("[BLE] 기존 연결 정리 완료")

        except Exception as e:
            log(
                f"[BLE] 연결 정리 실패: "
                f"{type(e).__name__}: {repr(e)}"
            )


    # =====================================================
    # 연결 재시도
    # =====================================================

    async def connect_with_retry(self):
        attempt = 1

        while True:
            log(
                f"[BLE] ===== 연결 시도 #{attempt} ====="
            )

            success = await self.connect()

            if success:
                log(
                    f"[BLE] 연결 시도 #{attempt} 성공"
                )
                return

            log(
                f"[BLE] 연결 시도 #{attempt} 실패"
            )

            log(
                f"[BLE] {RETRY_DELAY}초 후 재연결"
            )

            await asyncio.sleep(RETRY_DELAY)

            attempt += 1


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


    # =====================================================
    # BLE 패킷 전송
    # =====================================================

    async def send(self, packet):
        if not self.write_uuid:
            raise ValueError(
                "WRITE_CHARACTERISTIC_UUID가 "
                "설정되지 않았습니다."
            )


        # 연결이 끊어졌으면 재연결
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
                packet.encode("utf-8"),

                # ESP32 characteristic가 PROPERTY_WRITE이므로
                # Write Request 사용
                response=True
            )

            elapsed = (
                time.perf_counter() - start
            )

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

            if self.client is not None:
                log(
                    f"[BLE] 전송 실패 후 "
                    f"is_connected="
                    f"{self.client.is_connected}"
                )

            return False


    # =====================================================
    # ACK 초기화
    # =====================================================

    def clear_ack(self):
        self.ack_event.clear()

        log("[BLE] ACK event clear")


    # =====================================================
    # ACK 대기
    # =====================================================

    async def wait_for_ack(
        self,
        timeout=1.0
    ):
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


    # =====================================================
    # 프로그램 종료 시 연결 해제
    # =====================================================

    async def disconnect(self):
        if self.client is None:
            log(
                "[BLE] disconnect: client 없음"
            )
            return


        log(
            f"[BLE] disconnect 시작 "
            f"(is_connected="
            f"{self.client.is_connected})"
        )


        if not self.client.is_connected:
            log(
                "[BLE] 이미 연결 해제 상태"
            )
            return


        # Notify 종료
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


        # GATT 연결 해제
        try:
            log("[BLE] disconnect() 호출")

            await self.client.disconnect()

            log("[BLE] 연결 종료 완료")

        except Exception as e:
            log(
                f"[BLE] 연결 종료 중 오류: "
                f"{type(e).__name__}: {repr(e)}"
            )