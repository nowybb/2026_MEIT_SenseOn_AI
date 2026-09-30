from pathlib import Path
import pandas as pd

INPUT = Path("evaluation/ground_truth/kickboard_risk_events.csv")
OUTPUT = Path("evaluation/ground_truth/kickboard_events_for_merge.csv")

# 라벨링 간격
STEP_SEC = 0.2

df = pd.read_csv(INPUT)

events = []

for video, group in df.groupby("video"):

    group = group.sort_values("timestamp").reset_index(drop=True)

    danger = group[group["gt_level"] == "DANGER"]

    if danger.empty:
        print(f"{video}: DANGER 없음 → 전체 SAFE")
        continue

    event_id = 1
    start = None
    previous = None

    for timestamp in danger["timestamp"]:

        timestamp = float(timestamp)

        if start is None:
            start = timestamp
            previous = timestamp
            continue

        # 이전 DANGER와 연속되어 있지 않으면 이벤트 종료
        if timestamp - previous > STEP_SEC + 0.05:

            events.append({
                "video": video,
                "event_id": event_id,
                "start_sec": round(start, 2),
                "end_sec": round(previous, 2),
                "gt_warning": 1,
                "description": "킥보드 위험 상황",
            })

            event_id += 1
            start = timestamp

        previous = timestamp

    # 마지막 이벤트 저장
    events.append({
        "video": video,
        "event_id": event_id,
        "start_sec": round(start, 2),
        "end_sec": round(previous, 2),
        "gt_warning": 1,
        "description": "킥보드 위험 상황",
    })


result = pd.DataFrame(events)

result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 70)
print("KICKBOARD DANGER EVENTS")
print("=" * 70)

if result.empty:
    print("DANGER 이벤트 없음")
else:
    for video, group in result.groupby("video"):

        print()
        print(video)

        for _, row in group.iterrows():

            print(
                f"  event {row['event_id']}: "
                f"{row['start_sec']:.2f}s ~ "
                f"{row['end_sec']:.2f}s"
            )

print()
print(f"총 DANGER 이벤트: {len(result)}")
print(f"저장: {OUTPUT}")