import asyncio
import time
from pathlib import Path

import pandas as pd

from ble_sender import BLESender
from protocol import encode_hazard


# =========================
# 설정
# =========================
CSV_PATH = "realtest_risk_log.csv"

# 시연 결과 영상도 같이 재생하고 싶으면 경로 입력.
# 예: VIDEO_PATH = "outputs/realtest_risk_result.mp4"
# CSV → 진동만 테스트하려면 None 유지.
VIDEO_PATH = None

# realtest 영상 해상도가 1920x1080 기준일 때.
# 실제 영상 너비가 다르면 이 값만 수정.
FRAME_WIDTH = 1920

# UNKNOWN은 경고하지 않도록 SAFE로 처리
UNKNOWN_AS_SAFE = True

RISK_PRIORITY = {
    "UNKNOWN": 0,
    "SAFE": 1,
    "CAUTION": 2,
    "DANGER": 3,
}


def get_direction(center_x: float) -> str:
    """화면을 LEFT / CENTER / RIGHT 3등분."""
    left_boundary = FRAME_WIDTH / 3
    right_boundary = FRAME_WIDTH * 2 / 3

    if center_x < left_boundary:
        return "LEFT"
    elif center_x < right_boundary:
        return "CENTER"
    else:
        return "RIGHT"


def select_primary_hazard(frame_rows: pd.DataFrame):
    """
    같은 frame에 객체가 여러 개 있으면
    DANGER > CAUTION > SAFE > UNKNOWN 순으로 Primary Hazard 선정.
    같은 위험도라면 TTC가 더 짧은 객체를 우선.
    """
    if frame_rows.empty:
        return None

    rows = frame_rows.copy()

    rows["risk_priority"] = (
        rows["risk_level"]
        .fillna("UNKNOWN")
        .map(RISK_PRIORITY)
        .fillna(0)
    )

    # TTC가 없으면 매우 큰 값으로 처리
    rows["ttc_sort"] = pd.to_numeric(
        rows["visual_ttc"], errors="coerce"
    ).fillna(float("inf"))

    rows = rows.sort_values(
        by=["risk_priority", "ttc_sort"],
        ascending=[False, True]
    )

    return rows.iloc[0]


def row_to_hazard(row):
    risk = str(row["risk_level"]).upper()

    if UNKNOWN_AS_SAFE and risk == "UNKNOWN":
        risk = "SAFE"

    ttc = pd.to_numeric(
        pd.Series([row["visual_ttc"]]),
        errors="coerce"
    ).iloc[0]

    if pd.isna(ttc):
        ttc = None
    else:
        ttc = float(ttc)

    hazard = {
        "object": str(row["class_name"]),
        "direction": get_direction(float(row["center_x"])),
        "risk": risk,
        "ttc": ttc,
    }

    return hazard


async def send_safe(sender):
    """시연 종료 시 진동 확실히 OFF."""
    safe_hazard = {
        "object": "none",
        "direction": "CENTER",
        "risk": "SAFE",
        "ttc": None,
    }

    packet = encode_hazard(safe_hazard)
    await sender.send(packet)


async def run_csv_only(df, sender):
    """
    CSV의 timestamp를 그대로 따라가면서 BLE 전송.
    동일한 패킷은 다시 보내지 않고 상태가 바뀔 때만 전송.
    """
    grouped = list(df.groupby("frame", sort=True))

    start_real_time = time.perf_counter()
    first_timestamp = float(grouped[0][1]["timestamp"].iloc[0])

    last_packet = None

    for frame_no, frame_rows in grouped:
        timestamp = float(frame_rows["timestamp"].iloc[0])

        # CSV timestamp에 맞춰 실제 시간 동기화
        target_elapsed = timestamp - first_timestamp
        actual_elapsed = time.perf_counter() - start_real_time
        wait_time = target_elapsed - actual_elapsed

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        primary = select_primary_hazard(frame_rows)
        if primary is None:
            continue

        hazard = row_to_hazard(primary)
        packet = encode_hazard(hazard)

        # 같은 상태가 계속되면 중복 전송하지 않음
        if packet != last_packet:
            success = await sender.send(packet)

            print(
                f"[{timestamp:6.2f}s | frame {int(frame_no):3d}] "
                f"{hazard['object']:10s} "
                f"{hazard['direction']:6s} "
                f"{hazard['risk']:7s} "
                f"TTC={hazard['ttc']} "
                f"{'SEND OK' if success else 'SEND FAIL'}"
            )

            last_packet = packet


async def run_with_video(df, sender, video_path):
    """
    결과 영상을 재생하면서 현재 frame의 CSV 결과를 BLE로 전송.
    영상 화면에 보이는 위험도와 진동 타이밍을 맞추기 위한 모드.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # CSV frame → DataFrame 빠른 조회용
    frame_groups = {
        int(frame_no): rows
        for frame_no, rows in df.groupby("frame", sort=True)
    }

    last_packet = None
    frame_no = 1

    while True:
        frame_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_rows = frame_groups.get(frame_no)

        if frame_rows is not None:
            primary = select_primary_hazard(frame_rows)

            if primary is not None:
                hazard = row_to_hazard(primary)
                packet = encode_hazard(hazard)

                if packet != last_packet:
                    success = await sender.send(packet)

                    print(
                        f"[frame {frame_no:3d}] "
                        f"{hazard['object']:10s} "
                        f"{hazard['direction']:6s} "
                        f"{hazard['risk']:7s} "
                        f"TTC={hazard['ttc']} "
                        f"{'SEND OK' if success else 'SEND FAIL'}"
                    )

                    last_packet = packet

        cv2.imshow("SenseOn Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 원본 FPS에 맞춰 재생
        frame_duration = 1.0 / fps
        elapsed = time.perf_counter() - frame_start
        remain = frame_duration - elapsed

        if remain > 0:
            await asyncio.sleep(remain)

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()


async def main():
    csv_path = Path(CSV_PATH)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "frame",
        "timestamp",
        "class_name",
        "visual_ttc",
        "risk_level",
        "center_x",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV에 필요한 컬럼이 없습니다: {sorted(missing)}"
        )

    print(f"CSV 로드 완료: {len(df)} rows")
    print(
        f"frame {int(df['frame'].min())} ~ "
        f"{int(df['frame'].max())}"
    )
    print(df["risk_level"].value_counts())

    sender = BLESender()

    try:
        print("\nBLE 연결 중...")
        await sender.connect_with_retry()
        print("BLE 연결 완료\n")

        if VIDEO_PATH:
            await run_with_video(df, sender, VIDEO_PATH)
        else:
            await run_csv_only(df, sender)

    finally:
        print("\nSAFE 전송 → 진동 OFF")
        try:
            await send_safe(sender)
        except Exception as e:
            print("SAFE 전송 실패:", e)

        print("BLE 연결 종료")
        await sender.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
