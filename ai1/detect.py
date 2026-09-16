from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="videos/test.mp4",
    classes=[1, 2, 3],
    conf=0.3,
    stream=True
)

for result in results:

    boxes = result.boxes

    for box in boxes:

        # 객체 종류 번호
        class_id = int(box.cls[0])

        # 객체 이름
        class_name = model.names[class_id]

        # 탐지 confidence
        confidence = float(box.conf[0])

        # bounding box 좌표
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        # 중심 좌표
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # bbox 크기
        width = x2 - x1
        height = y2 - y1

        # bbox 면적
        area = width * height

        print("--------------------")
        print("class:", class_name)
        print("confidence:", round(confidence, 3))
        print("bbox:", x1, y1, x2, y2)
        print("center:", center_x, center_y)
        print("area:", area)