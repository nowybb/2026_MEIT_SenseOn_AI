from pathlib import Path
import csv
from collections import Counter, defaultdict


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

GT_DIR = BASE_DIR / "evaluation" / "ground_truth"

EVENTS_PATH = GT_DIR / "events.csv"
VIDEOS_PATH = GT_DIR / "videos.csv"

AI_LOG_DIR = (
    BASE_DIR
    / "ai1"
    / "outputs"
    / "trajectory_comparison"
)

WARNING_LEVELS = {
    "CAUTION",
    "DANGER",
}

# evaluate_ground_truth.py와 동일하게 사용
MAX_WARNING_GAP = 0.5


# ============================================================
# CSV
# ============================================================

def read_csv(path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(csv.DictReader(f))


# ============================================================
# 값 변환
# ============================================================

def to_bool(value):

    return str(value).strip().lower() == "true"


def to_float(value):

    value = str(value).strip()

    if not value:
        return None

    try:
        return float(value)

    except ValueError:
        return None


# ============================================================
# GT
# ============================================================

def load_gt_events():

    rows = read_csv(EVENTS_PATH)

    events = defaultdict(list)

    for row in rows:

        events[row["video"]].append({
            "start": float(row["start_sec"]),
            "end": float(row["end_sec"]),
        })

    return events


def load_complete_videos():

    rows = read_csv(VIDEOS_PATH)

    return [
        row["video"]
        for row in rows
        if row["label_status"].strip() == "complete"
    ]


def is_gt_danger(timestamp, events):

    return any(
        event["start"]
        <= timestamp
        <= event["end"]
        for event in events
    )


# ============================================================
# AI 로그
# ============================================================

def load_ai_rows(video_name):

    stem = Path(video_name).stem

    path = (
        AI_LOG_DIR
        / f"{stem}_risk_log.csv"
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"AI 로그가 없습니다: {path}"
        )

    rows = []

    for row in read_csv(path):

        rows.append({
            "timestamp": float(
                row["timestamp"]
            ),
            "track_id": int(
                row["track_id"]
            ),
            "class_name": row["class_name"],
            "risk": row["risk_level"].strip(),

            "approaching": to_bool(
                row["approaching"]
            ),

            "approach_rate": to_float(
                row["approach_rate"]
            ),

            "path_collision": to_bool(
                row["path_collision"]
            ),

            "bbox_path_collision": to_bool(
                row["bbox_path_collision"]
            ),

            "bbox_zone_overlap": to_float(
                row["bbox_zone_overlap"]
            ),

            "ttc": to_float(
                row["visual_ttc"]
            ),
        })

    return rows


# ============================================================
# SAFE 구간의 warning frame 추출
# ============================================================

def get_false_warning_rows(
    rows,
    gt_events,
):

    return [
        row
        for row in rows
        if (
            row["risk"]
            in WARNING_LEVELS
            and not is_gt_danger(
                row["timestamp"],
                gt_events,
            )
        )
    ]


# ============================================================
# frame 단위 warning을 event로 묶기
#
# timestamp 기준으로 먼저 묶고,
# 이벤트 안에 포함된 모든 객체 row를 보존한다.
# ============================================================

def build_false_alarm_events(rows):

    if not rows:
        return []

    rows = sorted(
        rows,
        key=lambda row: row["timestamp"],
    )

    # 동일 timestamp에 여러 객체가 있을 수 있음
    by_time = defaultdict(list)

    for row in rows:

        by_time[
            round(row["timestamp"], 4)
        ].append(row)

    times = sorted(by_time)

    events = []

    current_times = [times[0]]

    for timestamp in times[1:]:

        if (
            timestamp
            - current_times[-1]
            > MAX_WARNING_GAP
        ):

            events.append(
                make_event(
                    current_times,
                    by_time,
                )
            )

            current_times = [timestamp]

        else:

            current_times.append(
                timestamp
            )

    events.append(
        make_event(
            current_times,
            by_time,
        )
    )

    return events


def make_event(times, by_time):

    rows = []

    for timestamp in times:
        rows.extend(
            by_time[timestamp]
        )

    return {
        "start": times[0],
        "end": times[-1],
        "rows": rows,
    }


# ============================================================
# 이벤트 요약
# ============================================================

def summarize_event(event):

    rows = event["rows"]

    track_counts = Counter(
        row["track_id"]
        for row in rows
    )

    main_track = (
        track_counts.most_common(1)[0][0]
    )

    main_rows = [
        row
        for row in rows
        if row["track_id"] == main_track
    ]

    classes = Counter(
        row["class_name"]
        for row in main_rows
    )

    main_class = (
        classes.most_common(1)[0][0]
    )

    risks = Counter(
        row["risk"]
        for row in main_rows
    )

    approach_values = [
        row["approach_rate"]
        for row in main_rows
        if row["approach_rate"] is not None
    ]

    ttc_values = [
        row["ttc"]
        for row in main_rows
        if row["ttc"] is not None
    ]

    center_collision = any(
        row["path_collision"]
        for row in main_rows
    )

    bbox_collision = any(
        row["bbox_path_collision"]
        for row in main_rows
    )

    approaching = any(
        row["approaching"]
        for row in main_rows
    )

    return {
        "start": event["start"],
        "end": event["end"],
        "track_id": main_track,
        "class_name": main_class,
        "risks": risks,
        "approaching": approaching,

        "approach_min": (
            min(approach_values)
            if approach_values
            else None
        ),

        "approach_max": (
            max(approach_values)
            if approach_values
            else None
        ),

        "center_collision": center_collision,
        "bbox_collision": bbox_collision,

        "ttc_min": (
            min(ttc_values)
            if ttc_values
            else None
        ),

        "ttc_max": (
            max(ttc_values)
            if ttc_values
            else None
        ),
    }


# ============================================================
# 출력 보조
# ============================================================

def format_range(
    minimum,
    maximum,
    digits=3,
):

    if minimum is None:
        return "None"

    return (
        f"{minimum:.{digits}f}"
        f" ~ "
        f"{maximum:.{digits}f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    gt_by_video = load_gt_events()

    videos = load_complete_videos()

    all_events = []

    print()
    print("=" * 70)
    print("SenseOn False Alarm Analysis")
    print("=" * 70)

    for video_name in videos:

        rows = load_ai_rows(
            video_name
        )

        false_rows = (
            get_false_warning_rows(
                rows,
                gt_by_video.get(
                    video_name,
                    [],
                ),
            )
        )

        events = (
            build_false_alarm_events(
                false_rows
            )
        )

        print()
        print(
            f"{video_name}: "
            f"{len(events)} false alarm event(s)"
        )

        for index, event in enumerate(
            events,
            start=1,
        ):

            summary = summarize_event(
                event
            )

            all_events.append({
                "video": video_name,
                **summary,
            })

            print()
            print(
                f"  [FP {index:02d}]"
            )

            print(
                f"  time          : "
                f"{summary['start']:.2f}"
                f" ~ "
                f"{summary['end']:.2f}s"
            )

            print(
                f"  track_id      : "
                f"{summary['track_id']}"
            )

            print(
                f"  class         : "
                f"{summary['class_name']}"
            )

            print(
                f"  risk          : "
                f"{dict(summary['risks'])}"
            )

            print(
                f"  approaching   : "
                f"{summary['approaching']}"
            )

            print(
                "  approach rate : "
                + format_range(
                    summary["approach_min"],
                    summary["approach_max"],
                )
            )

            print(
                f"  center coll.  : "
                f"{summary['center_collision']}"
            )

            print(
                f"  bbox coll.    : "
                f"{summary['bbox_collision']}"
            )

            print(
                "  TTC           : "
                + format_range(
                    summary["ttc_min"],
                    summary["ttc_max"],
                    digits=2,
                )
            )

    # ========================================================
    # 전체 원인 통계
    # ========================================================

    total = len(all_events)

    center_true = sum(
        event["center_collision"]
        for event in all_events
    )

    bbox_true = sum(
        event["bbox_collision"]
        for event in all_events
    )

    center_only = sum(
        (
            event["center_collision"]
            and not event["bbox_collision"]
        )
        for event in all_events
    )

    ttc_danger = sum(
        (
            event["ttc_min"] is not None
            and event["ttc_min"] <= 2.0
        )
        for event in all_events
    )

    ttc_caution = sum(
        (
            event["ttc_min"] is not None
            and 2.0
            < event["ttc_min"]
            <= 4.0
        )
        for event in all_events
    )

    ttc_none = sum(
        event["ttc_min"] is None
        for event in all_events
    )

    tracks = Counter(
        (
            event["video"],
            event["track_id"],
        )
        for event in all_events
    )

    repeated_track_events = sum(
        count
        for count in tracks.values()
        if count > 1
    )

    print()
    print("=" * 70)
    print("FALSE ALARM SUMMARY")
    print("=" * 70)

    print(
        f"False alarm events       : "
        f"{total}"
    )

    print(
        f"Center collision=True    : "
        f"{center_true}/{total}"
    )

    print(
        f"BBox collision=True      : "
        f"{bbox_true}/{total}"
    )

    print(
        f"Center=True / BBox=False : "
        f"{center_only}/{total}"
    )

    print(
        f"TTC <= 2.0               : "
        f"{ttc_danger}/{total}"
    )

    print(
        f"2.0 < TTC <= 4.0         : "
        f"{ttc_caution}/{total}"
    )

    print(
        f"TTC=None                 : "
        f"{ttc_none}/{total}"
    )

    print(
        f"Repeated-track FP events : "
        f"{repeated_track_events}/{total}"
    )


if __name__ == "__main__":
    main()