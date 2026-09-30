from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 경로
# ============================================================

GT_DIR = Path("evaluation/ground_truth")

EVENTS_PATH = GT_DIR / "events.csv"
VIDEOS_PATH = GT_DIR / "videos.csv"

KICKBOARD_GT_PATH = (
    GT_DIR / "kickboard_risk_events.csv"
)

NORMAL_LOG_DIR = Path(
    "ai1/outputs/trajectory_comparison"
)

KICKBOARD_LOG_DIR = Path(
    "ai1/outputs/kickboard_risk"
)

RESULT_DIR = Path(
    "evaluation/results/all_risk"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STEP_SEC = 0.2

KICKBOARD_VIDEOS = {
    f"test{i}.mp4"
    for i in range(6, 15)
}


# ============================================================
# AI 로그
# ============================================================

def load_ai_log(video):

    stem = Path(video).stem

    if video in KICKBOARD_VIDEOS:
        path = (
            KICKBOARD_LOG_DIR
            / f"{stem}_risk_log.csv"
        )
    else:
        path = (
            NORMAL_LOG_DIR
            / f"{stem}_risk_log.csv"
        )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df.empty:
        return df

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    return df.dropna(
        subset=["timestamp"]
    )


def get_ai_level(
    log,
    timestamp,
    tolerance=0.15
):

    if log is None or log.empty:
        return "SAFE"

    nearby = log[
        (
            log["timestamp"]
            >= timestamp - tolerance
        )
        &
        (
            log["timestamp"]
            <= timestamp + tolerance
        )
    ]

    if nearby.empty:
        return "SAFE"

    risks = (
        nearby["risk_level"]
        .astype(str)
        .str.upper()
    )

    # 이번 평가는 DANGER만 양성
    if (risks == "DANGER").any():
        return "DANGER"

    return "SAFE"


# ============================================================
# 기존 영상 GT
# events.csv -> 0.2초 SAFE/DANGER
# ============================================================

def build_normal_gt():

    events = pd.read_csv(
        EVENTS_PATH
    )

    videos = pd.read_csv(
        VIDEOS_PATH
    )

    rows = []

    for _, video_row in videos.iterrows():

        video = str(
            video_row["video"]
        ).strip()

        # 킥보드는 별도 GT 사용
        if video in KICKBOARD_VIDEOS:
            continue

        try:
            duration = float(
                video_row["duration_sec"]
            )
        except (TypeError, ValueError):
            print(
                f"[SKIP] {video}: "
                "duration_sec 없음"
            )
            continue

        video_events = events[
            events["video"] == video
        ]

        timestamp = 0.0

        while timestamp <= duration:

            danger = False

            for _, event in video_events.iterrows():

                start = float(
                    event["start_sec"]
                )

                end = float(
                    event["end_sec"]
                )

                if (
                    start
                    <= timestamp
                    <= end
                ):
                    danger = True
                    break

            rows.append({
                "video": video,
                "timestamp": round(
                    timestamp,
                    2
                ),
                "gt_level": (
                    "DANGER"
                    if danger
                    else "SAFE"
                ),
                "group": "GENERAL",
            })

            timestamp += STEP_SEC

    return pd.DataFrame(rows)


# ============================================================
# 킥보드 GT
# ============================================================

def build_kickboard_gt():

    df = pd.read_csv(
        KICKBOARD_GT_PATH
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["gt_level"] = (
        df["gt_level"]
        .astype(str)
        .str.upper()
    )

    df = df[
        df["video"].isin(
            KICKBOARD_VIDEOS
        )
    ].copy()

    df["group"] = "KICKBOARD"

    return df[
        [
            "video",
            "timestamp",
            "gt_level",
            "group",
        ]
    ]


# ============================================================
# Confusion Matrix 계산
# ============================================================

def calculate_metrics(df):

    TP = len(df[
        (df["gt_level"] == "DANGER")
        &
        (df["ai_level"] == "DANGER")
    ])

    FP = len(df[
        (df["gt_level"] == "SAFE")
        &
        (df["ai_level"] == "DANGER")
    ])

    FN = len(df[
        (df["gt_level"] == "DANGER")
        &
        (df["ai_level"] == "SAFE")
    ])

    TN = len(df[
        (df["gt_level"] == "SAFE")
        &
        (df["ai_level"] == "SAFE")
    ])

    precision = (
        TP / (TP + FP)
        if TP + FP
        else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    accuracy = (
        (TP + TN)
        / len(df)
        if len(df)
        else 0
    )

    specificity = (
        TN / (TN + FP)
        if TN + FP
        else 0
    )

    return {
        "samples": len(df),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
    }


# ============================================================
# 메인
# ============================================================

def main():

    print("=" * 75)
    print("SENSEON ALL RISK MODEL EVALUATION")
    print("=" * 75)

    # --------------------------------------------------------
    # GT 통합
    # --------------------------------------------------------

    normal_gt = build_normal_gt()
    kickboard_gt = build_kickboard_gt()

    gt = pd.concat(
        [
            normal_gt,
            kickboard_gt,
        ],
        ignore_index=True
    )

    print()
    print(
        f"GENERAL GT : {len(normal_gt)}"
    )

    print(
        f"KICKBOARD GT: {len(kickboard_gt)}"
    )

    print(
        f"TOTAL GT   : {len(gt)}"
    )

    # --------------------------------------------------------
    # AI 결과 매칭
    # --------------------------------------------------------

    results = []

    for video in gt["video"].unique():

        video_gt = gt[
            gt["video"] == video
        ]

        ai_log = load_ai_log(
            video
        )

        if ai_log is None:
            print(
                f"[WARNING] AI 로그 없음: "
                f"{video}"
            )
            continue

        for _, row in video_gt.iterrows():

            timestamp = float(
                row["timestamp"]
            )

            ai_level = get_ai_level(
                ai_log,
                timestamp
            )

            results.append({
                "video": video,
                "group": row["group"],
                "timestamp": timestamp,
                "gt_level": row["gt_level"],
                "ai_level": ai_level,
                "correct": (
                    row["gt_level"]
                    == ai_level
                ),
            })

    result = pd.DataFrame(
        results
    )

    if result.empty:
        print(
            "평가 가능한 데이터가 없습니다."
        )
        return

    # --------------------------------------------------------
    # 전체 성능
    # --------------------------------------------------------

    overall = calculate_metrics(
        result
    )

    # --------------------------------------------------------
    # 그룹별 성능
    # --------------------------------------------------------

    group_rows = []

    for group, df in result.groupby(
        "group"
    ):

        metrics = calculate_metrics(
            df
        )

        metrics["group"] = group

        group_rows.append(
            metrics
        )

    group_metrics = pd.DataFrame(
        group_rows
    )

    # --------------------------------------------------------
    # 영상별 성능
    # --------------------------------------------------------

    video_rows = []

    for video, df in result.groupby(
        "video"
    ):

        metrics = calculate_metrics(
            df
        )

        metrics["video"] = video
        metrics["group"] = (
            df["group"].iloc[0]
        )

        video_rows.append(
            metrics
        )

    video_metrics = pd.DataFrame(
        video_rows
    )

    # 영상 번호 순서 정렬
    def video_sort(name):

        if name.startswith("test"):
            number = (
                name
                .replace("test", "")
                .replace(".mp4", "")
            )

            if number.isdigit():
                return (
                    0,
                    int(number)
                )

        if name == "test_receding.mp4":
            return (1, 0)

        if name == "realtest.mp4":
            return (2, 0)

        return (3, name)

    video_metrics["_sort"] = (
        video_metrics["video"]
        .apply(video_sort)
    )

    video_metrics = (
        video_metrics
        .sort_values("_sort")
        .drop(columns="_sort")
    )

    # --------------------------------------------------------
    # CSV 저장
    # --------------------------------------------------------

    result.to_csv(
        RESULT_DIR
        / "all_frame_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    video_metrics.to_csv(
        RESULT_DIR
        / "all_video_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    group_metrics.to_csv(
        RESULT_DIR
        / "group_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        [overall]
    ).to_csv(
        RESULT_DIR
        / "overall_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 그래프 1
    # 전체 성능
    # ========================================================

    metric_names = [
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "Specificity",
    ]

    metric_values = [
        overall["precision"] * 100,
        overall["recall"] * 100,
        overall["f1"] * 100,
        overall["accuracy"] * 100,
        overall["specificity"] * 100,
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        metric_names,
        metric_values
    )

    plt.ylim(
        0,
        100
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.title(
        "SenseOn Overall Risk Model Performance"
    )

    for bar, value in zip(
        bars,
        metric_values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "overall_metrics.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # 그래프 2
    # 영상별 F1
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    values = (
        video_metrics["f1"]
        * 100
    )

    bars = plt.bar(
        video_metrics["video"],
        values
    )

    plt.ylim(
        0,
        100
    )

    plt.ylabel(
        "F1 Score (%)"
    )

    plt.title(
        "F1 Score by Video"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.0f}",
            ha="center",
            fontsize=8
        )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "video_f1.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # 그래프 3
    # GENERAL vs KICKBOARD
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    x = np.arange(
        len(group_metrics)
    )

    width = 0.25

    plt.bar(
        x - width,
        group_metrics["precision"] * 100,
        width,
        label="Precision"
    )

    plt.bar(
        x,
        group_metrics["recall"] * 100,
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        group_metrics["f1"] * 100,
        width,
        label="F1"
    )

    plt.xticks(
        x,
        group_metrics["group"]
    )

    plt.ylim(
        0,
        100
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.title(
        "Performance by Object Group"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "group_comparison.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # 그래프 4
    # Confusion Matrix
    # ========================================================

    matrix = np.array([
        [
            overall["TN"],
            overall["FP"]
        ],
        [
            overall["FN"],
            overall["TP"]
        ],
    ])

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.imshow(
        matrix
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels([
        "Pred SAFE",
        "Pred DANGER"
    ])

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels([
        "Actual SAFE",
        "Actual DANGER"
    ])

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=18
            )

    ax.set_title(
        "SenseOn Risk Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR
        / "confusion_matrix.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # 터미널 표
    # ========================================================

    print()
    print("=" * 105)
    print("VIDEO PERFORMANCE")
    print("=" * 105)

    print(
        f"{'VIDEO':20s}"
        f"{'GROUP':12s}"
        f"{'TP':>5s}"
        f"{'FP':>5s}"
        f"{'FN':>5s}"
        f"{'TN':>5s}"
        f"{'PREC':>9s}"
        f"{'RECALL':>9s}"
        f"{'F1':>9s}"
        f"{'ACC':>9s}"
    )

    print("-" * 105)

    for _, row in video_metrics.iterrows():

        print(
            f"{row['video']:20s}"
            f"{row['group']:12s}"
            f"{int(row['TP']):5d}"
            f"{int(row['FP']):5d}"
            f"{int(row['FN']):5d}"
            f"{int(row['TN']):5d}"
            f"{row['precision']*100:8.1f}%"
            f"{row['recall']*100:8.1f}%"
            f"{row['f1']*100:8.1f}%"
            f"{row['accuracy']*100:8.1f}%"
        )

    print()
    print("=" * 75)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 75)

    print(
        f"Samples     : "
        f"{overall['samples']}"
    )

    print(
        f"TP / FP     : "
        f"{overall['TP']} / "
        f"{overall['FP']}"
    )

    print(
        f"FN / TN     : "
        f"{overall['FN']} / "
        f"{overall['TN']}"
    )

    print()

    print(
        f"Precision   : "
        f"{overall['precision']*100:.2f}%"
    )

    print(
        f"Recall      : "
        f"{overall['recall']*100:.2f}%"
    )

    print(
        f"F1 Score    : "
        f"{overall['f1']*100:.2f}%"
    )

    print(
        f"Accuracy    : "
        f"{overall['accuracy']*100:.2f}%"
    )

    print(
        f"Specificity : "
        f"{overall['specificity']*100:.2f}%"
    )

    print()
    print(
        "결과 저장:",
        RESULT_DIR
    )


if __name__ == "__main__":
    main()