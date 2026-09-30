from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

GT_PATH = Path(
    "evaluation/ground_truth/kickboard_risk_events.csv"
)

LOG_DIR = Path(
    "ai1/outputs/kickboard_risk"
)

RESULT_DIR = Path(
    "evaluation/results/kickboard_risk"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIDEOS = [
    f"test{i}.mp4"
    for i in range(6, 15)
]


def load_ai_log(video_name):

    stem = Path(video_name).stem

    path = LOG_DIR / f"{stem}_risk_log.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df.empty:
        return df

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    return df.dropna(subset=["timestamp"])


def get_ai_level(log, timestamp, tolerance=0.15):
    """
    GT timestamp에 가장 가까운 AI 결과를 찾는다.

    같은 시점에 객체가 여러 개라면
    DANGER가 하나라도 있으면 DANGER로 판단한다.

    CAUTION / SAFE / UNKNOWN / 탐지 없음은
    이번 이진 평가에서 SAFE로 처리한다.
    """

    if log is None or log.empty:
        return "SAFE"

    nearby = log[
        (log["timestamp"] >= timestamp - tolerance)
        &
        (log["timestamp"] <= timestamp + tolerance)
    ]

    if nearby.empty:
        return "SAFE"

    risks = (
        nearby["risk_level"]
        .astype(str)
        .str.upper()
        .tolist()
    )

    if "DANGER" in risks:
        return "DANGER"

    return "SAFE"


def main():

    gt = pd.read_csv(GT_PATH)

    gt["timestamp"] = pd.to_numeric(
        gt["timestamp"],
        errors="coerce"
    )

    gt["gt_level"] = (
        gt["gt_level"]
        .astype(str)
        .str.upper()
    )

    gt = gt[
        gt["video"].isin(VIDEOS)
    ].copy()

    results = []

    print("=" * 70)
    print("KICKBOARD SAFE / DANGER EVALUATION")
    print("=" * 70)

    for video in VIDEOS:

        video_gt = gt[
            gt["video"] == video
        ].copy()

        ai_log = load_ai_log(video)

        if video_gt.empty:
            print(f"\n{video}: GT 없음")
            continue

        if ai_log is None:
            print(f"\n{video}: AI 로그 없음")
            continue

        for _, row in video_gt.iterrows():

            timestamp = float(
                row["timestamp"]
            )

            gt_level = row["gt_level"]

            ai_level = get_ai_level(
                ai_log,
                timestamp
            )

            results.append({
                "video": video,
                "timestamp": timestamp,
                "gt_level": gt_level,
                "ai_level": ai_level,
                "correct": (
                    gt_level == ai_level
                )
            })

    result = pd.DataFrame(results)

    if result.empty:
        print("평가 가능한 데이터가 없습니다.")
        return

    # ----------------------------------------
    # Confusion Matrix
    # ----------------------------------------

    TP = len(result[
        (result["gt_level"] == "DANGER")
        &
        (result["ai_level"] == "DANGER")
    ])

    FP = len(result[
        (result["gt_level"] == "SAFE")
        &
        (result["ai_level"] == "DANGER")
    ])

    FN = len(result[
        (result["gt_level"] == "DANGER")
        &
        (result["ai_level"] == "SAFE")
    ])

    TN = len(result[
        (result["gt_level"] == "SAFE")
        &
        (result["ai_level"] == "SAFE")
    ])

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    accuracy = (
        (TP + TN)
        / (TP + FP + FN + TN)
    )

    # ----------------------------------------
    # 영상별 결과
    # ----------------------------------------

    video_rows = []

    for video in VIDEOS:

        v = result[
            result["video"] == video
        ]

        if v.empty:
            continue

        v_tp = len(v[
            (v["gt_level"] == "DANGER")
            &
            (v["ai_level"] == "DANGER")
        ])

        v_fp = len(v[
            (v["gt_level"] == "SAFE")
            &
            (v["ai_level"] == "DANGER")
        ])

        v_fn = len(v[
            (v["gt_level"] == "DANGER")
            &
            (v["ai_level"] == "SAFE")
        ])

        v_tn = len(v[
            (v["gt_level"] == "SAFE")
            &
            (v["ai_level"] == "SAFE")
        ])

        v_precision = (
            v_tp / (v_tp + v_fp)
            if v_tp + v_fp > 0
            else 0
        )

        v_recall = (
            v_tp / (v_tp + v_fn)
            if v_tp + v_fn > 0
            else 0
        )

        v_f1 = (
            2 * v_precision * v_recall
            / (v_precision + v_recall)
            if v_precision + v_recall > 0
            else 0
        )

        v_accuracy = (
            (v_tp + v_tn) / len(v)
        )

        video_rows.append({
            "video": video,
            "samples": len(v),
            "TP": v_tp,
            "FP": v_fp,
            "FN": v_fn,
            "TN": v_tn,
            "precision": v_precision,
            "recall": v_recall,
            "f1": v_f1,
            "accuracy": v_accuracy,
        })

    video_metrics = pd.DataFrame(
        video_rows
    )

    # ----------------------------------------
    # 저장
    # ----------------------------------------

    result.to_csv(
        RESULT_DIR / "frame_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    video_metrics.to_csv(
        RESULT_DIR / "video_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = pd.DataFrame([{
        "samples": len(result),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }])

    summary.to_csv(
        RESULT_DIR / "summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------------------
    # Confusion Matrix 그림
    # ----------------------------------------

    matrix = [
        [TN, FP],
        [FN, TP],
    ]

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    im = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Pred SAFE",
        "Pred DANGER"
    ])

    ax.set_yticks([0, 1])
    ax.set_yticklabels([
        "Actual SAFE",
        "Actual DANGER"
    ])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
                fontsize=16
            )

    ax.set_title(
        "Kickboard Risk Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # ----------------------------------------
    # 출력
    # ----------------------------------------

    print()
    print("-" * 70)
    print("VIDEO RESULTS")
    print("-" * 70)

    for _, row in video_metrics.iterrows():

        print(
            f"{row['video']:12s} | "
            f"TP {int(row['TP']):3d} | "
            f"FP {int(row['FP']):3d} | "
            f"FN {int(row['FN']):3d} | "
            f"TN {int(row['TN']):3d} | "
            f"F1 {row['f1'] * 100:6.2f}% | "
            f"ACC {row['accuracy'] * 100:6.2f}%"
        )

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(f"Samples   : {len(result)}")
    print(f"TP / FP   : {TP} / {FP}")
    print(f"FN / TN   : {FN} / {TN}")
    print()
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"Accuracy  : {accuracy * 100:.2f}%")

    print()
    print(
        "Results:",
        RESULT_DIR
    )


if __name__ == "__main__":
    main()
    