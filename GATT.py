import asyncio
from bleak import BleakClient

ADDRESS = "38:3E:51:C0:48:A6"

async def main():
    print("연결 시도")

    client = BleakClient(
        ADDRESS,
        timeout=20.0
    )

    try:
        await client.connect()

        print("is_connected =", client.is_connected)

        if client.is_connected:
            print("GATT 연결 성공")

    except Exception as e:
        print("GATT 연결 실패:", repr(e))

    finally:
        if client.is_connected:
            await client.disconnect()

asyncio.run(main())
