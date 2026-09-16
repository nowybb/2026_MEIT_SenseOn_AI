from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.track(
    source="videos/test.mp4",
    tracker="botsort.yaml",
    classes=[1, 2, 3],
    conf=0.3,
    persist=True,
    stream=True
)

for result in results:

    boxes = result.boxes

    if boxes.id is None:
        continue

    for box in boxes:

        track_id = int(box.id[0])

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(float, box.xyxy[0])

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        width = x2 - x1
        height = y2 - y1
        area = width * height

        print(
            f"ID={track_id} "
            f"class={class_name} "
            f"center=({center_x:.1f}, {center_y:.1f}) "
            f"area={area:.1f}"
        )