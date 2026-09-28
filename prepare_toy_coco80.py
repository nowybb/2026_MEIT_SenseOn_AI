from pathlib import Path
import shutil
import yaml

from ultralytics import YOLO


SRC = Path("toy_car_dataset/roboflow").resolve()
DST = Path("toy_car_dataset/coco80_toy").resolve()

MODEL_PATH = Path("ai1/yolo11n.pt")


# ------------------------------------------------------------
# 1. 원본 Roboflow 데이터 복사
# ------------------------------------------------------------

if DST.exists():
    raise RuntimeError(
        f"{DST} 폴더가 이미 있습니다. "
        "기존 결과를 확인한 뒤 삭제하고 다시 실행하세요."
    )

shutil.copytree(SRC, DST)

print(f"[COPY] {SRC}")
print(f"    -> {DST}")


# ------------------------------------------------------------
# 2. toy car class 0 -> COCO car class 2
# ------------------------------------------------------------

converted = 0

for split in ["train", "valid", "test"]:

    label_dir = DST / split / "labels"

    if not label_dir.exists():
        continue

    for label_path in label_dir.glob("*.txt"):

        new_lines = []

        for line in label_path.read_text().splitlines():

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            class_id = int(parts[0])

            if class_id != 0:
                raise ValueError(
                    f"{label_path}: 예상하지 못한 class_id={class_id}"
                )

            # Roboflow car=0 -> COCO car=2
            parts[0] = "2"

            new_lines.append(" ".join(parts))
            converted += 1

        label_path.write_text(
            "\n".join(new_lines)
            + ("\n" if new_lines else "")
        )


# ------------------------------------------------------------
# 3. 기존 YOLO11n의 COCO class 목록 가져오기
# ------------------------------------------------------------

model = YOLO(str(MODEL_PATH))

names = dict(model.names)

if names[2] != "car":
    raise RuntimeError(
        f"COCO class 2가 car가 아닙니다: {names[2]}"
    )


# ------------------------------------------------------------
# 4. 새로운 80-class data.yaml 생성
# ------------------------------------------------------------

data = {
    "path": str(DST),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(names),
    "names": names,
}

yaml_path = DST / "data_coco80.yaml"

with open(yaml_path, "w") as f:
    yaml.safe_dump(
        data,
        f,
        sort_keys=False,
        allow_unicode=True,
    )


print()
print(f"[DONE] {converted} annotations converted")
print("[CLASS] toy car: 0 -> 2 (COCO car)")
print(f"[YAML] {yaml_path}")
print(f"[CLASSES] {len(names)}")