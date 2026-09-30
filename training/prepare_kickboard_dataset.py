from pathlib import Path
import json
import shutil

# ============================================================
# SenseOn Kickboard Dataset Builder
#
# AI-Hub 개인형 이동장치 안전 데이터
# PM_code 28,29,30,31,32,33,35,36 -> kickboard(class 0)
# ============================================================

DATA_ROOT = Path(
    "/Users/yeonwoo/Desktop/120.개인형 이동장치 안전 데이터/01.데이터"
)

TRAIN_IMAGE_ROOT = DATA_ROOT / "1.Training/원천데이터"
TRAIN_LABEL_ROOT = DATA_ROOT / "1.Training/라벨링데이터"

VAL_IMAGE_ROOT = DATA_ROOT / "2.Validation/원천데이터"
VAL_LABEL_ROOT = DATA_ROOT / "2.Validation/라벨링데이터"

PROJECT_ROOT = Path(
    "/Users/yeonwoo/Desktop/2026_MEIT_SenseOn_AI"
)

OUTPUT_ROOT = PROJECT_ROOT / "training/datasets/kickboard"

KICKBOARD_CODES = {
    "28", "29", "30", "31",
    "32", "33", "35", "36"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG"
}


def prepare_directories():

    for split in ["train", "val"]:
        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


def build_image_index(image_root):

    print(f"\n이미지 검색 중: {image_root}")

    index = {}

    for path in image_root.rglob("*"):

        if path.suffix in IMAGE_EXTENSIONS:
            index[path.stem] = path

    print(f"발견 이미지: {len(index):,}장")

    return index


def find_pm_list(data):

    annotations = data.get("annotations", {})

    # 일반적인 구조
    if isinstance(annotations, dict):
        pm = annotations.get("PM", [])

        if isinstance(pm, list):
            return pm

    # 혹시 annotations가 list 구조인 경우까지 대응
    if isinstance(annotations, list):

        result = []

        for item in annotations:

            if not isinstance(item, dict):
                continue

            pm = item.get("PM", [])

            if isinstance(pm, list):
                result.extend(pm)

        return result

    return []


def convert_bbox(points, image_width, image_height):

    if not isinstance(points, (list, tuple)):
        return None

    if len(points) != 4:
        return None

    try:
        left = float(points[0])
        bottom = float(points[1])
        width = float(points[2])
        height = float(points[3])
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    # AI-Hub 문서의 points:
    # [left, bottom, width, height]
    #
    # 실제 JSON의 좌표 체계를 그대로 YOLO bbox로 변환.
    # 여기서 bottom 필드가 이미지 좌상단 기준 y 위치로
    # 저장되어 있는지 이후 샘플 시각화로 반드시 검증한다.

    x_center = (left + width / 2) / image_width
    y_center = (bottom + height / 2) / image_height

    w = width / image_width
    h = height / image_height

    # YOLO 범위 보호
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return x_center, y_center, w, h


def process_split(image_root, label_root, split):

    print("\n" + "=" * 65)
    print(f"{split.upper()} 변환 시작")
    print("=" * 65)

    image_index = build_image_index(image_root)

    json_files = list(label_root.rglob("*.json"))
    json_files += list(label_root.rglob("*.JSON"))

    print(f"발견 JSON: {len(json_files):,}개")

    output_image_dir = OUTPUT_ROOT / "images" / split
    output_label_dir = OUTPUT_ROOT / "labels" / split

    kickboard_images = 0
    kickboard_objects = 0

    no_image = 0
    invalid_json = 0
    invalid_bbox = 0

    code_counts = {}

    for i, json_path in enumerate(json_files, 1):

        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

        except Exception:
            invalid_json += 1
            continue

        pm_list = find_pm_list(data)

        kickboards = []

        for obj in pm_list:

            if not isinstance(obj, dict):
                continue

            code = str(obj.get("PM_code", "")).strip()

            if code in KICKBOARD_CODES:
                kickboards.append(obj)

        # 킥보드가 없는 이미지는 사용하지 않음
        if not kickboards:
            continue

        image_path = image_index.get(json_path.stem)

        if image_path is None:
            no_image += 1
            continue

        description = data.get("description", {})

        try:
            image_width = float(description["imageWidth"])
            image_height = float(description["imageHeight"])
        except (KeyError, TypeError, ValueError):
            print(
                f"\n[경고] 이미지 크기 정보 없음: "
                f"{json_path.name}"
            )
            continue

        yolo_lines = []

        for obj in kickboards:

            if obj.get("shape_type") != "bbox":
                continue

            bbox = convert_bbox(
                obj.get("points"),
                image_width,
                image_height
            )

            if bbox is None:
                invalid_bbox += 1
                continue

            x, y, w, h = bbox

            # 모든 킥보드 유형을 class 0으로 통합
            yolo_lines.append(
                f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )

            code = str(obj.get("PM_code", "")).strip()

            code_counts[code] = (
                code_counts.get(code, 0) + 1
            )

        if not yolo_lines:
            continue

        # 파일명이 겹칠 가능성을 줄이기 위해
        # 원본 stem 사용
        destination_image = (
            output_image_dir / image_path.name
        )

        destination_label = (
            output_label_dir / f"{image_path.stem}.txt"
        )

        if not destination_image.exists():
            shutil.copy2(
                image_path,
                destination_image
            )

        with open(
            destination_label,
            "w",
            encoding="utf-8"
        ) as f:
            f.write("\n".join(yolo_lines))

        kickboard_images += 1
        kickboard_objects += len(yolo_lines)

        if i % 10000 == 0:
            print(
                f"진행: {i:,}/{len(json_files):,} "
                f"| 킥보드 이미지 {kickboard_images:,}"
            )

    print("\n--- 결과 ---")
    print(f"킥보드 이미지 : {kickboard_images:,}")
    print(f"킥보드 객체   : {kickboard_objects:,}")
    print(f"이미지 미매칭 : {no_image:,}")
    print(f"JSON 오류     : {invalid_json:,}")
    print(f"BBOX 오류     : {invalid_bbox:,}")

    print("\nPM_code별 객체 수")

    for code in sorted(code_counts, key=int):
        print(
            f"  {code}: {code_counts[code]:,}"
        )

    return kickboard_images, kickboard_objects


def create_yaml():

    yaml_path = OUTPUT_ROOT / "data.yaml"

    content = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val

names:
  0: kickboard
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(content)

    print(f"\ndata.yaml 생성: {yaml_path}")


def main():

    prepare_directories()

    train_result = process_split(
        TRAIN_IMAGE_ROOT,
        TRAIN_LABEL_ROOT,
        "train"
    )

    val_result = process_split(
        VAL_IMAGE_ROOT,
        VAL_LABEL_ROOT,
        "val"
    )

    create_yaml()

    print("\n" + "=" * 65)
    print("DATASET BUILD COMPLETE")
    print("=" * 65)

    print(
        f"TRAIN : {train_result[0]:,} images / "
        f"{train_result[1]:,} objects"
    )

    print(
        f"VAL   : {val_result[0]:,} images / "
        f"{val_result[1]:,} objects"
    )

    print(f"\nDataset: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()