"""Raspberry Pi BLE Central -> ESP32 GATT Write, Notify ACK 확인.

하드웨어 및 실제 MTU 협상은 별도 검증해야 한다.
"""

import asyncio

DEVICE_NAME = "SenseOn_ESP32"
SERVICE_UUID = "7f4d1000-2e3b-4f7a-9b2a-1c2d3e4f5001"
WRITE_UUID = "7f4d1001-2e3b-4f7a-9b2a-1c2d3e4f5001"
NOTIFY_UUID = "7f4d1002-2e3b-4f7a-9b2a-1c2d3e4f5001"


class ESP32BLE:
    def __init__(self, address=None, ack_timeout=1.0):
        self.address = address
        self.ack_timeout = ack_timeout
        self.client = None
        self._ack = asyncio.Event()

    def _on_notify(self, _sender, data):
        if bytes(data).decode("utf-8", errors="replace").strip() == "ACK":
            self._ack.set()

    async def connect(self):
        # 테스트 모드에서는 이 함수를 호출하지 않으므로 bleak 설치가 불필요하다.
        from bleak import BleakClient, BleakScanner

        if self.client is not None and self.client.is_connected:
            return
        await self.close()
        if self.address:
            device = await BleakScanner.find_device_by_address(self.address, timeout=10.0)
        else:
            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
        if device is None:
            raise ConnectionError("ESP32 BLE 장치를 찾지 못했습니다")

        self.client = BleakClient(device, timeout=15.0)
        try:
            await self.client.connect()
            # UUID가 실제로 존재하는지 연결 시점에 검사한다.
            if self.client.services.get_characteristic(WRITE_UUID) is None:
                raise RuntimeError("ESP32 WRITE characteristic을 찾을 수 없습니다")
            if self.client.services.get_characteristic(NOTIFY_UUID) is None:
                raise RuntimeError("ESP32 NOTIFY characteristic을 찾을 수 없습니다")
            await self.client.start_notify(NOTIFY_UUID, self._on_notify)
            print("[BLE] 연결 성공 / WRITE, NOTIFY 확인")
            print("[BLE] 20바이트 초과 패킷은 MTU 협상 및 실기기 검증 필요")
        except Exception:
            await self.close()
            raise

    async def send(self, packet):
        if not isinstance(packet, str):
            raise TypeError("BLE 패킷은 문자열이어야 합니다")
        if self.client is None or not self.client.is_connected:
            await self.connect()
        payload = packet.encode("utf-8")
        if len(payload) > 20:
            print(f"[BLE MTU 주의] {len(payload)}바이트: 실기기에서 긴 Write 지원 확인 필요")
        self._ack.clear()  # 이전 패킷의 ACK를 새 패킷에 사용하지 않음
        try:
            await self.client.write_gatt_char(WRITE_UUID, payload, response=True)
            await asyncio.wait_for(self._ack.wait(), timeout=self.ack_timeout)
        except Exception:
            await self.close()
            raise
        print(f"[BLE TX/ACK] {packet}")

    async def close(self):
        client, self.client = self.client, None
        if client is not None:
            try:
                if client.is_connected:
                    await client.disconnect()
            except Exception as exc:
                print(f"[BLE] 연결 해제 중 오류: {exc}")
