from ultralytics import YOLO
from collections import defaultdict, deque
from pathlib import Path
import cv2


# ============================================================
# 1. 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# 한 번에 분석할 영상 목록
VIDEO_NAMES = [
    "test1.mp4",
    "test2.mp4",
    "test3.mp4",
    "test4.mp4",
    "test5.mp4",
    "test_receding.mp4"
]

HISTORY_SIZE = 10
APPROACH_THRESHOLD = 0.05

MODEL_PATH = BASE_DIR / "yolo11n.pt"


# ============================================================
# 2. 영상 하나를 분석하는 함수
# ============================================================

def process_video(video_name):

    video_path = VIDEO_DIR / video_name

    output_path = (
        OUTPUT_DIR / f"{video_path.stem}_result.mp4"
    )

    print("\n===================================")
    print("분석 시작:", video_name)
    print("===================================")

    # 영상 존재 여부 확인
    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상이 없습니다: {video_path}"
        )

    # --------------------------------------------------------
    # 원본 영상 정보
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise ValueError("FPS 오류")

    if width <= 0 or height <= 0:
        raise ValueError("영상 크기 오류")

    print("FPS:", fps)
    print("Frame size:", width, "x", height)

    # --------------------------------------------------------
    # 결과 영상 저장 설정
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("결과 영상 생성 실패")

    frame_index = 0

    try:

        # ----------------------------------------------------
        # YOLO + BoT-SORT
        # 영상별로 모델을 새로 생성하여 추적 상태 분리
        # ----------------------------------------------------

        model = YOLO(str(MODEL_PATH))

        results = model.track(
            source=str(video_path),
            tracker="botsort.yaml",
            classes=[1, 2, 3],
            conf=0.3,
            persist=True,
            stream=True
        )

        # 영상별 Track History 초기화
        track_history = defaultdict(
            lambda: deque(maxlen=HISTORY_SIZE)
        )

        # ----------------------------------------------------
        # 프레임별 분석
        # ----------------------------------------------------

        for result in results:

            frame_index += 1

            frame = result.orig_img.copy()

            boxes = result.boxes

            if boxes is not None and boxes.id is not None:

                for box in boxes:

                    # ----------------------------------------
                    # 객체 정보
                    # ----------------------------------------

                    track_id = int(box.id[0])

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = map(
                        float, box.xyxy[0]
                    )

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    width_box = x2 - x1
                    height_box = y2 - y1

                    area = width_box * height_box

                    # ----------------------------------------
                    # Track History 저장
                    # ----------------------------------------

                    track_history[track_id].append({
                        "frame": frame_index,
                        "center_x": center_x,
                        "center_y": center_y,
                        "area": area
                    })

                    history = track_history[track_id]

                    # ----------------------------------------
                    # 접근 여부 계산
                    # ----------------------------------------

                    approach_rate = 0.0
                    state = "UNKNOWN"

                    if len(history) >= 2:

                        old = history[0]
                        current = history[-1]

                        old_area = old["area"]
                        current_area = current["area"]

                        frame_gap = (
                            current["frame"] - old["frame"]
                        )

                        if old_area > 0 and frame_gap > 0:

                            time_gap = frame_gap / fps

                            area_change_ratio = (
                                current_area - old_area
                            ) / old_area

                            approach_rate = (
                                area_change_ratio / time_gap
                            )

                            if approach_rate > APPROACH_THRESHOLD:
                                state = "APPROACHING"

                            else:
                                state = "NOT APPROACHING"

                    # ----------------------------------------
                    # 색상 결정
                    # ----------------------------------------

                    if state == "APPROACHING":
                        color = (0, 0, 255)

                    elif state == "NOT APPROACHING":
                        color = (0, 255, 0)

                    else:
                        color = (0, 255, 255)

                    # ----------------------------------------
                    # Bounding Box
                    # ----------------------------------------

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2
                    )

                    # 중심점
                    cv2.circle(
                        frame,
                        (int(center_x), int(center_y)),
                        5,
                        color,
                        -1
                    )

                    # ----------------------------------------
                    # 텍스트
                    # ----------------------------------------

                    label1 = (
                        f"ID:{track_id} "
                        f"{class_name} "
                        f"Conf:{confidence:.2f}"
                    )

                    label2 = (
                        f"Rate:{approach_rate:.2f} "
                        f"{state}"
                    )

                    cv2.putText(
                        frame,
                        label1,
                        (int(x1), max(int(y1) - 25, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        label2,
                        (int(x1), max(int(y1) - 7, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            # -----------------------------------------------
            # 프레임 번호 표시
            # -----------------------------------------------

            cv2.putText(
                frame,
                f"Frame: {frame_index}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # 결과 영상 저장
            writer.write(frame)

    finally:
        writer.release()

    if frame_index == 0:
        raise RuntimeError("처리된 프레임이 없습니다.")

    print("분석 완료:", video_name)
    print("처리 프레임:", frame_index)
    print("저장 위치:", output_path)

    return output_path


# ============================================================
# 3. 전체 영상 순차 실행
# ============================================================

success = []
failed = []

for video_name in VIDEO_NAMES:

    try:

        output_path = process_video(video_name)

        success.append(video_name)

    except Exception as e:

        print("오류 발생:", video_name)
        print("오류 내용:", e)

        failed.append(video_name)


# ============================================================
# 4. 최종 결과 요약
# ============================================================

print("\n===================================")
print("전체 영상 분석 종료")
print("===================================")

print("성공:", len(success))
print("실패:", len(failed))

print("\n성공 목록:")
for name in success:
    print("-", name)

print("\n실패 목록:")
for name in failed:
    print("-", name)

print("===================================")