import asyncio
import time
import pandas as pd

from ble_sender import BLESender
from protocol import encode_hazard


CSV_PATH = "test_video_log.csv"

RISK_PRIORITY = {
    "UNKNOWN": 0,
    "SAFE": 1,
    "CAUTION": 2,
    "DANGER": 3,
}


def load_timeline():
    df = pd.read_csv(CSV_PATH)

    timeline = []

    # 같은 timestamp에 탐지된 모든 객체를 묶음
    for timestamp, rows in df.groupby("timestamp", sort=True):

        risks = (
            rows["risk_level"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .tolist()
        )

        # 해당 시점에서 가장 위험한 상태 하나 선택
        risk = max(
            risks,
            key=lambda x: RISK_PRIORITY.get(x, 0)
        )

        # UNKNOWN만 존재하면 진동하지 않음
        if risk == "UNKNOWN":
            risk = "SAFE"

        timeline.append(
            (float(timestamp), risk)
        )

    return timeline


async def send_risk(sender, risk):

    hazard = {
        "object": "car",
        "direction": "CENTER",
        "risk": risk,
        "ttc": None
    }

    packet = encode_hazard(hazard)

    success = await sender.send(packet)

    return success


async def main():

    timeline = load_timeline()

    # 먼저 CSV를 어떻게 해석했는지 출력
    print("===== CSV 위험도 timeline =====")

    previous = None

    for timestamp, risk in timeline:

        if risk != previous:
            print(f"{timestamp:.3f}s -> {risk}")
            previous = risk

    print("===============================\n")

    sender = BLESender()

    print("BLE 연결 중...")
    await sender.connect_with_retry()
    print("BLE 연결 완료")

    start = time.perf_counter()

    last_risk = None

    try:

        for timestamp, risk in timeline:

            # CSV timestamp까지 정확히 대기
            while True:

                elapsed = time.perf_counter() - start
                remaining = timestamp - elapsed

                if remaining <= 0:
                    break

                await asyncio.sleep(
                    min(remaining, 0.01)
                )

            # 상태가 달라질 때만 ESP32로 전송
            if risk != last_risk:

                success = await send_risk(
                    sender,
                    risk
                )

                print(
                    f"[{timestamp:.3f}s] "
                    f"{last_risk} -> {risk} "
                    f"{'SEND OK' if success else 'SEND FAIL'}"
                )

                last_risk = risk

        # CSV 종료 후 진동 OFF
        if last_risk != "SAFE":

            await send_risk(
                sender,
                "SAFE"
            )

            print("END -> SAFE")

    finally:

        await sender.disconnect()

        print("BLE 연결 종료")


if __name__ == "__main__":
    asyncio.run(main())