import asyncio

from mock_ai import get_mock_hazard
from protocol import encode_hazard
from ble_sender import BLESender


async def main():

    sender = BLESender()

    try:

        # BLE 연결은 처음에 한 번만
        await sender.connect_with_retry()

        print("BLE 연결 완료")


        while True:

            # Mock AI 데이터 생성
            hazard = get_mock_hazard()

            packet = encode_hazard(hazard)


            print("Mock AI:", hazard)

            print("Packet:", packet)


            # 이전 ACK 초기화
            sender.clear_ack()


            # ESP32로 전송
            send_success = await sender.send(packet)


            if not send_success:

                print("BLE 전송 실패")

                await asyncio.sleep(1)

                continue


            print("BLE 전송 성공")


            # ACK 대기
            ack_received = await sender.wait_for_ack()


            if ack_received:

                print("ACK 수신 성공")

            else:

                print("ACK 수신 실패")


            # 다음 데이터 보내기 전 잠깐 대기
            await asyncio.sleep(1)


    except KeyboardInterrupt:

        print("사용자가 종료했습니다.")


    finally:

        await sender.disconnect()

        print("BLE 연결 종료")


if __name__ == "__main__":
    asyncio.run(main())