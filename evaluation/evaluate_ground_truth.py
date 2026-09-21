from pathlib import Path
import csv


# ============================================================
# 경로
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

RESULT_DIR = BASE_DIR / "evaluation" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = RESULT_DIR / "ground_truth_evaluation.csv"


# CAUTION 또는 DANGER면 실제 경고로 간주
WARNING_LEVELS = {
    "CAUTION",
    "DANGER",
}


# ============================================================
# CSV 읽기
# ============================================================

def read_csv(path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# GT 읽기
# ============================================================

def load_gt_events():

    rows = read_csv(EVENTS_PATH)

    events = {}

    for row in rows:

        video = row["video"]

        events.setdefault(
            video,
            []
        )

        events[video].append({
            "event_id": int(row["event_id"]),
            "start": float(row["start_sec"]),
            "end": float(row["end_sec"]),
        })

    return events


def load_videos():

    rows = read_csv(VIDEOS_PATH)

    videos = {}

    for row in rows:

        if row["label_status"].strip() != "complete":
            continue

        duration_text = row["duration_sec"].strip()

        duration = (
            float(duration_text)
            if duration_text
            else None
        )

        videos[row["video"]] = {
            "duration": duration,
            "notes": row["notes"],
        }

    return videos


# ============================================================
# AI 로그 읽기
# ============================================================

def load_ai_log(video_name):

    stem = Path(video_name).stem

    path = (
        AI_LOG_DIR
        / f"{stem}_risk_log.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"AI 로그가 없습니다: {path}"
        )

    rows = read_csv(path)

    detections = []

    for row in rows:

        risk = row["risk_level"].strip()

        detections.append({
            "timestamp": float(row["timestamp"]),
            "track_id": int(row["track_id"]),
            "risk": risk,
        })

    return detections


# ============================================================
# 특정 시각에 GT 위험 여부
# ============================================================

def is_gt_danger(timestamp, gt_events):

    for event in gt_events:

        if (
            event["start"]
            <= timestamp
            <= event["end"]
        ):
            return True

    return False


# ============================================================
# AI warning timestamp 추출
#
# 같은 프레임에 여러 객체가 있어도
# 하나라도 CAUTION/DANGER면 시스템 warning
# ============================================================

def get_warning_timestamps(detections):

    warning_times = set()

    for row in detections:

        if row["risk"] in WARNING_LEVELS:

            # 동일 프레임 근처 중복 제거
            warning_times.add(
                round(
                    row["timestamp"],
                    4,
                )
            )

    return sorted(
        warning_times
    )


# ============================================================
# 연속 AI 경고를 하나의 warning event로 묶기
# ============================================================

def build_warning_events(
    warning_times,
    max_gap=0.5,
):

    if not warning_times:
        return []

    events = []

    start = warning_times[0]
    previous = warning_times[0]

    for timestamp in warning_times[1:]:

        if timestamp - previous > max_gap:

            events.append({
                "start": start,
                "end": previous,
            })

            start = timestamp

        previous = timestamp

    events.append({
        "start": start,
        "end": previous,
    })

    return events


# ============================================================
# 두 시간 구간 overlap
# ============================================================

def overlaps(
    start1,
    end1,
    start2,
    end2,
):

    return (
        start1 <= end2
        and start2 <= end1
    )


# ============================================================
# 영상 하나 평가
# ============================================================

def evaluate_video(
    video_name,
    video_info,
    gt_events,
):

    detections = load_ai_log(
        video_name
    )

    warning_times = (
        get_warning_timestamps(
            detections
        )
    )

    warning_events = (
        build_warning_events(
            warning_times
        )
    )

    # --------------------------------------------------------
    # 영상 길이
    # --------------------------------------------------------

    duration = video_info["duration"]

    if duration is None:

        if detections:
            duration = max(
                row["timestamp"]
                for row in detections
            )
        else:
            duration = 0.0

    # --------------------------------------------------------
    # GT event Recall
    # --------------------------------------------------------

    tp = 0
    fn = 0

    delays = []

    for gt in gt_events:

        warnings_inside = [
            timestamp
            for timestamp in warning_times
            if (
                gt["start"]
                <= timestamp
                <= gt["end"]
            )
        ]

        if warnings_inside:

            tp += 1

            first_warning = min(
                warnings_inside
            )

            delay = (
                first_warning
                - gt["start"]
            )

            delays.append(
                delay
            )

        else:
            fn += 1

    # --------------------------------------------------------
    # False Alarm event
    #
    # GT 위험구간과 전혀 겹치지 않는 AI warning event
    # --------------------------------------------------------

    false_alarm_events = []

    for warning in warning_events:

        matched = False

        for gt in gt_events:

            if overlaps(
                warning["start"],
                warning["end"],
                gt["start"],
                gt["end"],
            ):
                matched = True
                break

        if not matched:
            false_alarm_events.append(
                warning
            )

    # --------------------------------------------------------
    # GT SAFE 시간 계산
    # --------------------------------------------------------

    danger_duration = sum(
        max(
            0.0,
            event["end"]
            - event["start"],
        )
        for event in gt_events
    )

    safe_duration = max(
        0.0,
        duration - danger_duration,
    )

    safe_minutes = (
        safe_duration / 60.0
    )

    if safe_minutes > 0:

        false_alarms_per_min = (
            len(false_alarm_events)
            / safe_minutes
        )

    else:
        false_alarms_per_min = None

    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    return {
        "video": video_name,
        "duration_sec": duration,
        "gt_events": len(gt_events),
        "tp": tp,
        "fn": fn,
        "warning_events": len(
            warning_events
        ),
        "false_alarm_events": len(
            false_alarm_events
        ),
        "safe_duration_sec": safe_duration,
        "false_alarms_per_min": (
            false_alarms_per_min
        ),
        "delays": delays,
    }


# ============================================================
# 전체 평가
# ============================================================

def main():

    videos = load_videos()
    gt_by_video = load_gt_events()

    results = []

    total_gt = 0
    total_tp = 0
    total_fn = 0

    total_false_alarms = 0
    total_safe_duration = 0.0

    all_delays = []

    print()
    print("=" * 65)
    print("SenseOn Ground Truth Evaluation")
    print("=" * 65)

    for video_name, video_info in videos.items():

        gt_events = gt_by_video.get(
            video_name,
            [],
        )

        result = evaluate_video(
            video_name,
            video_info,
            gt_events,
        )

        results.append(
            result
        )

        total_gt += result["gt_events"]
        total_tp += result["tp"]
        total_fn += result["fn"]

        total_false_alarms += (
            result["false_alarm_events"]
        )

        total_safe_duration += (
            result["safe_duration_sec"]
        )

        all_delays.extend(
            result["delays"]
        )

        print()
        print(video_name)
        print(
            f"  GT events      : "
            f"{result['gt_events']}"
        )
        print(
            f"  TP / FN        : "
            f"{result['tp']} / "
            f"{result['fn']}"
        )
        print(
            f"  AI warnings    : "
            f"{result['warning_events']}"
        )
        print(
            f"  False alarms   : "
            f"{result['false_alarm_events']}"
        )

        if result["delays"]:

            print(
                "  Warning delay  : "
                + ", ".join(
                    f"{delay:.2f}s"
                    for delay
                    in result["delays"]
                )
            )

    # ========================================================
    # 전체 Recall
    # ========================================================

    recall = (
        total_tp / total_gt
        if total_gt > 0
        else None
    )

    safe_minutes = (
        total_safe_duration / 60.0
    )

    false_alarm_rate = (
        total_false_alarms
        / safe_minutes
        if safe_minutes > 0
        else None
    )

    average_delay = (
        sum(all_delays)
        / len(all_delays)
        if all_delays
        else None
    )

    print()
    print("=" * 65)
    print("TOTAL")
    print("=" * 65)

    print(
        f"GT 위험 이벤트 : "
        f"{total_gt}"
    )

    print(
        f"TP             : "
        f"{total_tp}"
    )

    print(
        f"FN             : "
        f"{total_fn}"
    )

    if recall is not None:

        print(
            f"Recall         : "
            f"{recall * 100:.2f}%"
        )

    print(
        f"False alarms   : "
        f"{total_false_alarms}"
    )

    print(
        f"GT SAFE time   : "
        f"{total_safe_duration:.2f}s"
    )

    if false_alarm_rate is not None:

        print(
            f"False alarms/min: "
            f"{false_alarm_rate:.2f}"
        )

    if average_delay is not None:

        print(
            f"Average delay  : "
            f"{average_delay:.2f}s"
        )

    # ========================================================
    # 결과 CSV 저장
    # ========================================================

    with RESULT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "video",
            "duration_sec",
            "gt_events",
            "tp",
            "fn",
            "warning_events",
            "false_alarm_events",
            "safe_duration_sec",
            "false_alarms_per_min",
            "warning_delays_sec",
        ])

        for result in results:

            writer.writerow([
                result["video"],
                round(
                    result["duration_sec"],
                    3,
                ),
                result["gt_events"],
                result["tp"],
                result["fn"],
                result["warning_events"],
                result["false_alarm_events"],
                round(
                    result["safe_duration_sec"],
                    3,
                ),
                (
                    round(
                        result[
                            "false_alarms_per_min"
                        ],
                        3,
                    )
                    if result[
                        "false_alarms_per_min"
                    ] is not None
                    else ""
                ),
                ";".join(
                    f"{delay:.3f}"
                    for delay
                    in result["delays"]
                ),
            ])

    print()
    print(
        "결과 저장:",
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()