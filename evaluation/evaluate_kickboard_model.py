from pathlib import Path
import csv
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# 설정
# ============================================================

MODEL_PATH = Path("ai1/kickboard_best.pt")
VIDEO_DIR = Path("ai1/videos")

GT_CSV = Path(
    "evaluation/ground_truth/kickboard_events.csv"
)

OUTPUT_DIR = Path(
    "evaluation/results/kickboard"
)

CONF_THRESHOLD = 0.25
IMG_SIZE = 640


# ============================================================
# Ground Truth 읽기
# ============================================================

def load_ground_truth():

    rows = []

    with GT_CSV.open(
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append({
                "video": row["video"],
                "frame": int(row["frame"]),
                "gt": int(row["kickboard"]),
            })

    return rows


# ============================================================
# 안전한 나눗셈
# ============================================================

def safe_div(a, b):

    if b == 0:
        return 0.0

    return a / b


# ============================================================
# 평가
# ============================================================

def evaluate():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("Kickboard Detection Evaluation")
    print("=" * 65)

    print(f"Model : {MODEL_PATH}")
    print(f"GT    : {GT_CSV}")
    print(f"Conf  : {CONF_THRESHOLD}")
    print()

    model = YOLO(str(MODEL_PATH))

    ground_truth = load_ground_truth()

    videos = sorted(
        set(
            row["video"]
            for row in ground_truth
        )
    )

    frame_results = []

    # --------------------------------------------------------
    # 영상별 처리
    # --------------------------------------------------------

    for video_name in videos:

        video_path = VIDEO_DIR / video_name

        if not video_path.exists():

            print(
                f"[WARNING] 영상 없음: "
                f"{video_path}"
            )

            continue

        gt_rows = [
            row
            for row in ground_truth
            if row["video"] == video_name
        ]

        print()
        print("-" * 65)
        print(
            f"{video_name} "
            f"({len(gt_rows)} labelled frames)"
        )
        print("-" * 65)

        cap = cv2.VideoCapture(
            str(video_path)
        )

        for index, row in enumerate(gt_rows, 1):

            frame_number = row["frame"]
            gt = row["gt"]

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            ok, frame = cap.read()

            if not ok:

                print(
                    f"[FRAME ERROR] "
                    f"{video_name}: "
                    f"{frame_number}"
                )

                continue

            # -----------------------------------------------
            # YOLO inference
            # -----------------------------------------------

            start = time.perf_counter()

            results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            result = results[0]

            confidences = []

            if result.boxes is not None:

                for box in result.boxes:

                    conf = float(
                        box.conf[0].item()
                    )

                    confidences.append(conf)

            # 하나라도 탐지되면 positive
            pred = 1 if len(confidences) > 0 else 0

            max_conf = (
                max(confidences)
                if confidences
                else 0.0
            )

            num_detections = len(confidences)

            # -----------------------------------------------
            # confusion matrix
            # -----------------------------------------------

            if gt == 1 and pred == 1:
                outcome = "TP"

            elif gt == 0 and pred == 1:
                outcome = "FP"

            elif gt == 1 and pred == 0:
                outcome = "FN"

            else:
                outcome = "TN"

            frame_results.append({
                "video": video_name,
                "frame": frame_number,
                "gt": gt,
                "prediction": pred,
                "outcome": outcome,
                "detections": num_detections,
                "max_confidence": max_conf,
                "inference_ms": elapsed_ms,
            })

            if (
                index % 10 == 0
                or index == len(gt_rows)
            ):

                print(
                    f"{index:>4}/"
                    f"{len(gt_rows)}"
                )

        cap.release()

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(frame_results)

    if df.empty:
        print("평가 결과가 없습니다.")
        return

    frame_csv = (
        OUTPUT_DIR /
        "kickboard_frame_results.csv"
    )

    df.to_csv(
        frame_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 전체 지표
    # ========================================================

    tp = int(
        (df["outcome"] == "TP").sum()
    )

    fp = int(
        (df["outcome"] == "FP").sum()
    )

    fn = int(
        (df["outcome"] == "FN").sum()
    )

    tn = int(
        (df["outcome"] == "TN").sum()
    )

    precision = safe_div(
        tp,
        tp + fp
    )

    recall = safe_div(
        tp,
        tp + fn
    )

    f1 = safe_div(
        2 * precision * recall,
        precision + recall
    )

    accuracy = safe_div(
        tp + tn,
        tp + fp + fn + tn
    )

    positive_predictions = df[
        df["prediction"] == 1
    ]

    avg_confidence = (
        positive_predictions[
            "max_confidence"
        ].mean()
        if not positive_predictions.empty
        else 0.0
    )

    avg_inference_ms = (
        df["inference_ms"].mean()
    )

    # ========================================================
    # 영상별 지표
    # ========================================================

    video_rows = []

    for video_name, group in df.groupby("video"):

        v_tp = int(
            (group["outcome"] == "TP").sum()
        )

        v_fp = int(
            (group["outcome"] == "FP").sum()
        )

        v_fn = int(
            (group["outcome"] == "FN").sum()
        )

        v_tn = int(
            (group["outcome"] == "TN").sum()
        )

        v_precision = safe_div(
            v_tp,
            v_tp + v_fp
        )

        v_recall = safe_div(
            v_tp,
            v_tp + v_fn
        )

        v_f1 = safe_div(
            2 * v_precision * v_recall,
            v_precision + v_recall
        )

        v_accuracy = safe_div(
            v_tp + v_tn,
            len(group)
        )

        video_rows.append({
            "video": video_name,
            "frames": len(group),
            "TP": v_tp,
            "FP": v_fp,
            "FN": v_fn,
            "TN": v_tn,
            "precision": v_precision,
            "recall": v_recall,
            "f1": v_f1,
            "accuracy": v_accuracy,
            "avg_inference_ms":
                group["inference_ms"].mean(),
        })

    video_df = pd.DataFrame(video_rows)

    video_csv = (
        OUTPUT_DIR /
        "kickboard_video_metrics.csv"
    )

    video_df.to_csv(
        video_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # Summary CSV
    # ========================================================

    summary = pd.DataFrame([
        {
            "labelled_frames": len(df),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_inference_ms": avg_inference_ms,
        }
    ])

    summary_csv = (
        OUTPUT_DIR /
        "kickboard_summary.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 그래프 1 : 전체 성능
    # ========================================================

    metric_names = [
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
    ]

    metric_values = [
        precision * 100,
        recall * 100,
        f1 * 100,
        accuracy * 100,
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        metric_names,
        metric_values
    )

    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title(
        "Kickboard Detection Performance"
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
        OUTPUT_DIR /
        "overall_metrics.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # 그래프 2 : 영상별 F1
    # ========================================================

    plt.figure(figsize=(10, 5))

    values = (
        video_df["f1"] * 100
    )

    bars = plt.bar(
        video_df["video"],
        values
    )

    plt.ylim(0, 100)
    plt.ylabel("F1 Score (%)")
    plt.title(
        "Kickboard Detection F1 by Video"
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
        OUTPUT_DIR /
        "video_f1.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # 그래프 3 : Confusion Matrix
    # ========================================================

    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    image = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Pred Negative",
        "Pred Positive"
    ])

    ax.set_yticklabels([
        "Actual Negative",
        "Actual Positive"
    ])

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=16
            )

    ax.set_title(
        "Kickboard Detection Confusion Matrix"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # 터미널 결과
    # ========================================================

    print()
    print("=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)

    print(
        f"Labelled frames : {len(df):,}"
    )

    print()
    print(
        f"TP / FP / FN / TN : "
        f"{tp} / {fp} / {fn} / {tn}"
    )

    print()
    print(
        f"Precision       : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall          : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score        : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Accuracy        : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Avg Confidence  : "
        f"{avg_confidence:.3f}"
    )

    print(
        f"Avg Inference   : "
        f"{avg_inference_ms:.2f} ms"
    )

    print()
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate()