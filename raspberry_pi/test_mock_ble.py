import asyncio

from mock_ai import get_mock_hazard
from protocol import encode_hazard
from ble_sender import BLESender


async def main():
    sender = BLESender()

    hazard = get_mock_hazard()
    packet = encode_hazard(hazard)

    print("Mock AI:", hazard)
    print("Packet:", packet)

    try:
        await sender.connect_with_retry()

        sender.clear_ack()

        send_success = await sender.send(packet)

        if not send_success:
            print("BLE 전송 실패")
            return

        print("BLE 전송 성공")

        ack_received = await sender.wait_for_ack()

        if ack_received:
            print("ACK 수신 성공")
        else:
            print("ACK 수신 실패")

    finally:
        await sender.disconnect()


if __name__ == "__main__":
    asyncio.run(main())