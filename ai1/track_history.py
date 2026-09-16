from ultralytics import YOLO
from collections import defaultdict, deque
import cv2
import os


# ============================================================
# 1. 설정
# ============================================================

VIDEO_PATH = "videos/test_approaching.mp4"
OUTPUT_PATH = "outputs/approaching_result.mp4"

# 최근 몇 프레임의 정보를 저장할지
HISTORY_SIZE = 10

# 접근 여부 판단 임시 기준값
APPROACH_THRESHOLD = 0.05

# 결과 영상 재생 FPS
# 영상이 너무 빠르게 재생되는 문제를 막기 위해 고정
OUTPUT_FPS = 23.0


# ============================================================
# 2. outputs 폴더 생성
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# 3. 원본 영상 정보 가져오기
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"영상을 열 수 없습니다: {VIDEO_PATH}")

original_fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()


print("===================================")
print("원본 영상 FPS:", original_fps)
print("결과 영상 FPS:", OUTPUT_FPS)
print("Frame size:", frame_width, "x", frame_height)
print("Frame count:", frame_count)
print("===================================")


# ============================================================
# 4. 결과 영상 저장 설정
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    OUTPUT_FPS,
    (frame_width, frame_height)
)

if not writer.isOpened():
    raise RuntimeError("결과 영상 파일을 생성할 수 없습니다.")


# ============================================================
# 5. YOLO 모델 불러오기
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 6. 객체별 Track History
# ============================================================

track_history = defaultdict(
    lambda: deque(maxlen=HISTORY_SIZE)
)


# ============================================================
# 7. YOLO + BoT-SORT Tracking 실행
# ============================================================

results = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",
    classes=[1, 2, 3],       # bicycle, car, motorcycle
    conf=0.3,
    persist=True,
    stream=True
)


frame_index = 0


# ============================================================
# 8. 프레임별 처리
# ============================================================

for result in results:

    frame_index += 1

    # 원본 프레임 복사
    frame = result.orig_img.copy()

    boxes = result.boxes


    # ========================================================
    # Tracking ID가 존재하는 객체만 처리
    # ========================================================

    if boxes.id is not None:

        for box in boxes:

            # ------------------------------------------------
            # 기본 객체 정보
            # ------------------------------------------------

            track_id = int(box.id[0])

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(float, box.xyxy[0])


            # ------------------------------------------------
            # bbox 중심 좌표
            # ------------------------------------------------

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2


            # ------------------------------------------------
            # bbox 크기 및 면적
            # ------------------------------------------------

            width = x2 - x1
            height = y2 - y1

            area = width * height


            # ------------------------------------------------
            # Track History 저장
            # ------------------------------------------------

            track_history[track_id].append({
                "frame": frame_index,
                "center_x": center_x,
                "center_y": center_y,
                "area": area
            })


            # ------------------------------------------------
            # ApproachRate 계산
            # ------------------------------------------------

            approach_rate = 0.0
            approaching = False

            history = track_history[track_id]


            # 최소 2개 이상의 기록이 있을 때 계산
            if len(history) >= 2:

                old = history[0]
                current = history[-1]

                old_area = old["area"]
                current_area = current["area"]

                frame_gap = (
                    current["frame"]
                    - old["frame"]
                )


                if (
                    old_area > 0
                    and frame_gap > 0
                    and original_fps > 0
                ):

                    # 실제 영상 시간 간격
                    time_gap = (
                        frame_gap / original_fps
                    )


                    # bbox 면적 변화 비율
                    area_change_ratio = (
                        current_area - old_area
                    ) / old_area


                    # 초당 bbox 면적 변화율
                    approach_rate = (
                        area_change_ratio / time_gap
                    )


                    # 접근 여부
                    if approach_rate > APPROACH_THRESHOLD:
                        approaching = True


            # ------------------------------------------------
            # 색상 및 상태 결정
            # ------------------------------------------------

            if approaching:

                # 빨강
                color = (0, 0, 255)

                state = "APPROACHING"

            else:

                # 초록
                color = (0, 255, 0)

                state = "NOT APPROACHING"


            # ------------------------------------------------
            # Bounding Box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )


            # ------------------------------------------------
            # 객체 중심점
            # ------------------------------------------------

            cv2.circle(
                frame,
                (int(center_x), int(center_y)),
                5,
                color,
                -1
            )


            # ------------------------------------------------
            # 첫 번째 표시 정보
            # ID / class / confidence
            # ------------------------------------------------

            label1 = (
                f"ID:{track_id} "
                f"{class_name} "
                f"Conf:{confidence:.2f}"
            )


            # ------------------------------------------------
            # 두 번째 표시 정보
            # ApproachRate / 상태
            # ------------------------------------------------

            label2 = (
                f"Rate:{approach_rate:.2f} "
                f"{state}"
            )


            # ------------------------------------------------
            # 텍스트 위치
            # ------------------------------------------------

            text_x = int(x1)

            text_y1 = max(
                int(y1) - 25,
                20
            )

            text_y2 = max(
                int(y1) - 7,
                40
            )


            # ------------------------------------------------
            # 텍스트 출력
            # ------------------------------------------------

            cv2.putText(
                frame,
                label1,
                (text_x, text_y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            cv2.putText(
                frame,
                label2,
                (text_x, text_y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )


    # ========================================================
    # 현재 프레임 번호 표시
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_index}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 결과 영상에 프레임 저장
    # ========================================================

    writer.write(frame)


# ============================================================
# 9. VideoWriter 종료
# ============================================================

writer.release()


# ============================================================
# 10. 완료 출력
# ============================================================

print()
print("===================================")
print("영상 분석 완료")
print("저장 위치:", OUTPUT_PATH)
print("===================================")