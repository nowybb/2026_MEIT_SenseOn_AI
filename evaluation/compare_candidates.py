from pathlib import Path
import csv
from collections import defaultdict

from ai2.risk import RiskStabilizer


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

MAX_GAP = 0.5


# ============================================================
# 비교 후보
# ============================================================

CANDIDATES = {
    "Baseline": "baseline_risk",
    "Stable Trajectory": "stable_trajectory_risk",
    "BBox Collision": "bbox_collision_risk",
    "TTC Required": "ttc_required_risk",
    "Stable + TTC": "stable_ttc_required_risk",
    "Stable + Stabilizer": "stable_stabilized_risk",
    "Stable + TTC + Stabilizer": "stable_ttc_stabilized_risk",
}


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
# GT
# ============================================================

def load_gt():

    events = defaultdict(list)

    for row in read_csv(EVENTS_PATH):

        events[row["video"]].append({
            "start": float(row["start_sec"]),
            "end": float(row["end_sec"]),
        })

    return events


def load_videos():

    videos = {}

    for row in read_csv(VIDEOS_PATH):

        if row["label_status"].strip() != "complete":
            continue

        duration = row["duration_sec"].strip()

        videos[row["video"]] = (
            float(duration)
            if duration
            else None
        )

    return videos


# ============================================================
# Risk Stabilizer
# ============================================================

def add_stabilized_risk(
    rows,
    source_column,
    output_column,
    release_frames=3,
):

    # 영상마다 새로운 stabilizer 사용
    stabilizer = RiskStabilizer(
        release_frames=release_frames
    )

    rows = sorted(
        rows,
        key=lambda row: (
            float(row["timestamp"]),
            int(row["track_id"]),
        ),
    )

    for row in rows:

        raw_risk = row[source_column].strip()

        if raw_risk not in {
            "SAFE",
            "CAUTION",
            "DANGER",
        }:
            row[output_column] = raw_risk
            continue

        track_id = int(
            row["track_id"]
        )

        row[output_column] = stabilizer.update(
            track_id,
            raw_risk,
        )

    return rows


# ============================================================
# AI 로그
# ============================================================

def load_log(video):

    stem = Path(video).stem

    path = (
        AI_LOG_DIR
        / f"{stem}_risk_log.csv"
    )

    rows = read_csv(path)

    # --------------------------------------------------------
    # Stable Trajectory + Stabilizer
    # --------------------------------------------------------

    rows = add_stabilized_risk(
        rows=rows,
        source_column="stable_trajectory_risk",
        output_column="stable_stabilized_risk",
        release_frames=3,
    )

    # --------------------------------------------------------
    # Stable + TTC Required + Stabilizer
    # --------------------------------------------------------

    rows = add_stabilized_risk(
        rows=rows,
        source_column="stable_ttc_required_risk",
        output_column="stable_ttc_stabilized_risk",
        release_frames=3,
    )

    return rows


# ============================================================
# GT 판정
# ============================================================

def in_gt_danger(timestamp, events):

    return any(
        event["start"]
        <= timestamp
        <= event["end"]
        for event in events
    )


# ============================================================
# Warning timestamp → Warning event
# ============================================================

def build_events(times):

    times = sorted(set(times))

    if not times:
        return []

    events = []

    start = times[0]
    previous = times[0]

    for timestamp in times[1:]:

        if timestamp - previous > MAX_GAP:

            events.append(
                (start, previous)
            )

            start = timestamp

        previous = timestamp

    events.append(
        (start, previous)
    )

    return events


# ============================================================
# 후보 평가
# ============================================================

def evaluate_candidate(
    column,
    videos,
    gt_by_video,
):

    total_gt = 0
    tp = 0
    fn = 0

    false_alarm_events = 0
    false_warning_time = 0.0
    safe_duration = 0.0

    delays = []

    per_video = []

    for video, duration in videos.items():

        rows = load_log(video)

        gt_events = gt_by_video.get(
            video,
            [],
        )

        # duration이 비어 있으면 AI 로그 마지막 timestamp 사용
        if duration is None:

            duration = max(
                float(row["timestamp"])
                for row in rows
            )

        total_gt += len(gt_events)

        # ====================================================
        # 시스템 Warning timestamp
        #
        # 같은 timestamp에 여러 객체가 있어도
        # 하나라도 CAUTION/DANGER이면 시스템 Warning
        # ====================================================

        warning_times = sorted(set(

            round(
                float(row["timestamp"]),
                4,
            )

            for row in rows

            if row[column].strip()
            in WARNING_LEVELS
        ))

        # ====================================================
        # GT Recall + Warning Delay
        # ====================================================

        video_tp = 0
        video_fn = 0
        video_delays = []

        for gt in gt_events:

            inside = [
                timestamp
                for timestamp in warning_times
                if (
                    gt["start"]
                    <= timestamp
                    <= gt["end"]
                )
            ]

            if inside:

                tp += 1
                video_tp += 1

                delay = (
                    min(inside)
                    - gt["start"]
                )

                delays.append(delay)
                video_delays.append(delay)

            else:

                fn += 1
                video_fn += 1

        # ====================================================
        # SAFE 구간 Warning
        # ====================================================

        safe_warning_times = [

            timestamp
            for timestamp in warning_times

            if not in_gt_danger(
                timestamp,
                gt_events,
            )
        ]

        fp_events = build_events(
            safe_warning_times
        )

        false_alarm_events += len(
            fp_events
        )

        # ====================================================
        # False Warning Time
        # ====================================================

        for start, end in fp_events:

            false_warning_time += max(
                0.0,
                end - start,
            )

        # ====================================================
        # SAFE 시간
        # ====================================================

        danger_duration = sum(

            max(
                0.0,
                gt["end"] - gt["start"],
            )

            for gt in gt_events
        )

        video_safe_duration = max(
            0.0,
            duration - danger_duration,
        )

        safe_duration += (
            video_safe_duration
        )

        per_video.append({
            "video": video,
            "tp": video_tp,
            "fn": video_fn,
            "fp": len(fp_events),
            "delays": video_delays,
        })

    # ========================================================
    # 전체 지표
    # ========================================================

    recall = (
        tp / total_gt
        if total_gt
        else 0.0
    )

    safe_minutes = (
        safe_duration / 60.0
    )

    fp_per_min = (
        false_alarm_events
        / safe_minutes
        if safe_minutes > 0
        else 0.0
    )

    false_warning_ratio = (
        false_warning_time
        / safe_duration
        if safe_duration > 0
        else 0.0
    )

    avg_delay = (
        sum(delays) / len(delays)
        if delays
        else None
    )

    return {
        "gt": total_gt,
        "tp": tp,
        "fn": fn,
        "recall": recall,

        "fp": false_alarm_events,
        "fp_per_min": fp_per_min,

        "false_warning_time": (
            false_warning_time
        ),

        "false_warning_ratio": (
            false_warning_ratio
        ),

        "avg_delay": avg_delay,
        "per_video": per_video,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    gt = load_gt()
    videos = load_videos()

    print()
    print("=" * 106)
    print("SenseOn GT Candidate Comparison")
    print("=" * 106)

    results = {}

    for name, column in CANDIDATES.items():

        results[name] = evaluate_candidate(
            column,
            videos,
            gt,
        )

    print()

    print(
        f"{'Candidate':<30}"
        f"{'Recall':>10}"
        f"{'TP/FN':>10}"
        f"{'FP':>8}"
        f"{'FP/min':>12}"
        f"{'FP Time':>12}"
        f"{'FP Time %':>12}"
        f"{'Delay':>12}"
    )

    print("-" * 106)

    # ========================================================
    # 전체 비교표
    # ========================================================

    for name, result in results.items():

        delay = (
            f"{result['avg_delay']:.3f}s"
            if result["avg_delay"] is not None
            else "-"
        )

        print(
            f"{name:<30}"
            f"{result['recall'] * 100:>9.2f}%"
            f"{result['tp']:>5}/"
            f"{result['fn']:<4}"
            f"{result['fp']:>8}"
            f"{result['fp_per_min']:>12.2f}"
            f"{result['false_warning_time']:>11.2f}s"
            f"{result['false_warning_ratio'] * 100:>11.2f}%"
            f"{delay:>12}"
        )

    # ========================================================
    # 영상별 상세
    # ========================================================

    for name, result in results.items():

        print()
        print("=" * 106)
        print(name)
        print("=" * 106)

        for video in result["per_video"]:

            delay_text = (
                ", ".join(
                    f"{value:.3f}s"
                    for value
                    in video["delays"]
                )
                if video["delays"]
                else "-"
            )

            print(
                f"{video['video']:<20}"
                f"TP={video['tp']} "
                f"FN={video['fn']} "
                f"FP={video['fp']} "
                f"delay={delay_text}"
            )


if __name__ == "__main__":
    main()