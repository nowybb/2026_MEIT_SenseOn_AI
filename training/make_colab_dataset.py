from pathlib import Path
from PIL import Image
import random
import shutil
import yaml

# ============================================================
# 설정
# ============================================================

PROJECT_ROOT = Path("/Users/yeonwoo/Desktop/2026_MEIT_SenseOn_AI")

SOURCE_ROOT = PROJECT_ROOT / "training/datasets/kickboard"
OUTPUT_ROOT = PROJECT_ROOT / "training/datasets/kickboard_colab"

TRAIN_COUNT = 10000
VAL_COUNT = 2000

MAX_SIZE = 640
JPEG_QUALITY = 90
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)


# ============================================================
# 이미지 축소
# ============================================================

def resize_and_save(src, dst):
    with Image.open(src) as img:
        img = img.convert("RGB")

        w, h = img.size

        scale = min(MAX_SIZE / w, MAX_SIZE / h, 1.0)

        new_w = round(w * scale)
        new_h = round(h * scale)

        if scale < 1.0:
            img = img.resize(
                (new_w, new_h),
                Image.Resampling.LANCZOS
            )

        dst = dst.with_suffix(".jpg")

        img.save(
            dst,
            "JPEG",
            quality=JPEG_QUALITY,
            optimize=True
        )

    return dst


# ============================================================
# split 생성
# ============================================================

def make_split(split, count):

    image_dir = SOURCE_ROOT / "images" / split
    label_dir = SOURCE_ROOT / "labels" / split

    out_image_dir = OUTPUT_ROOT / "images" / split
    out_label_dir = OUTPUT_ROOT / "labels" / split

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print()
    print("=" * 60)
    print(f"{split.upper()} 처리")
    print("=" * 60)

    print(f"원본 이미지: {len(images):,}")

    if len(images) < count:
        raise RuntimeError(
            f"{split}: 요청한 {count:,}장보다 "
            f"원본 이미지({len(images):,})가 적습니다."
        )

    # 항상 같은 데이터가 선택되도록 seed 고정
    selected = random.sample(images, count)

    success = 0
    missing_label = 0
    errors = 0

    for i, image_path in enumerate(selected, 1):

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_label += 1
            continue

        try:
            # 이미지는 640 이하로 축소
            resize_and_save(
                image_path,
                out_image_dir / f"{image_path.stem}.jpg"
            )

            # YOLO 좌표는 정규화되어 있으므로 그대로 복사
            shutil.copy2(
                label_path,
                out_label_dir / f"{image_path.stem}.txt"
            )

            success += 1

        except Exception as e:
            errors += 1
            print(f"\n오류: {image_path.name}")
            print(e)

        if i % 1000 == 0:
            print(
                f"{i:,}/{count:,} "
                f"| 완료 {success:,} "
                f"| 라벨없음 {missing_label:,} "
                f"| 오류 {errors:,}"
            )

    print()
    print("--- 결과 ---")
    print(f"완료      : {success:,}")
    print(f"라벨 없음 : {missing_label:,}")
    print(f"오류      : {errors:,}")

    return success


# ============================================================
# 실행
# ============================================================

if OUTPUT_ROOT.exists():
    print(f"기존 출력 폴더 삭제: {OUTPUT_ROOT}")
    shutil.rmtree(OUTPUT_ROOT)

train_count = make_split("train", TRAIN_COUNT)
val_count = make_split("val", VAL_COUNT)


# ============================================================
# Colab용 data.yaml
# ============================================================

yaml_data = {
    "path": "/content/kickboard",
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "kickboard"
    }
}

yaml_path = OUTPUT_ROOT / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        yaml_data,
        f,
        allow_unicode=True,
        sort_keys=False
    )


print()
print("=" * 60)
print("COLAB DATASET COMPLETE")
print("=" * 60)

print(f"TRAIN : {train_count:,}")
print(f"VAL   : {val_count:,}")
print(f"SIZE  : max {MAX_SIZE}px")
print(f"PATH  : {OUTPUT_ROOT}")
print(f"YAML  : {yaml_path}")