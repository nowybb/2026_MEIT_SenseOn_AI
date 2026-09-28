from pathlib import Path
import json
import random
import shutil
from collections import defaultdict, Counter
from PIL import Image


# =========================
# 설정
# =========================

IMAGE_ROOT = Path("/Users/yeonwoo/Desktop/CCTV2")
LABEL_ROOT = Path("/Users/yeonwoo/Desktop/CCTV2_labeled")

OUTPUT_ROOT = Path("training/datasets/kickboard_cctv2")

KICKBOARD_CODES = {"29", "30", "35", "36"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

VAL_RATIO = 0.2
SEED = 42


def convert_bbox(x, y, w, h, image_width, image_height):
    """AI-Hub [x, y, w, h] -> YOLO normalized bbox."""

    xc = (x + w / 2) / image_width
    yc = (y + h / 2) / image_height
    nw = w / image_width
    nh = h / image_height

    return xc, yc, nw, nh


def main():
    random.seed(SEED)

    # =========================
    # 이미지 / JSON 검색
    # =========================

    print("이미지 검색 중...")

    images = sorted(
        p for p in IMAGE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    print(f"전체 이미지: {len(images)}")

    print("JSON 인덱스 생성 중...")

    json_files = list(LABEL_ROOT.rglob("*.json"))

    json_map = {}
    duplicate_json_stems = set()

    for json_path in json_files:
        if json_path.stem in json_map:
            duplicate_json_stems.add(json_path.stem)

        json_map[json_path.stem] = json_path

    if duplicate_json_stems:
        print(f"경고: 중복 JSON stem {len(duplicate_json_stems)}개")

    # =========================
    # 통계
    # =========================

    groups = defaultdict(list)

    total_boxes = 0

    missing_json = 0
    broken_json = 0
    broken_image = 0

    invalid_points = 0
    invalid_size = 0

    fully_outside_boxes = 0
    clipped_boxes = 0

    metadata_size_mismatch = 0

    code_counter = Counter()

    # =========================
    # 이미지별 처리
    # =========================

    for idx, image_path in enumerate(images, 1):

        json_path = json_map.get(image_path.stem)

        if json_path is None:
            missing_json += 1
            continue

        # JSON 읽기
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception as e:
            broken_json += 1
            print(f"[JSON 오류] {json_path.name}: {e}")
            continue

        # 실제 이미지 크기 사용
        try:
            with Image.open(image_path) as img:
                width, height = img.size

        except Exception as e:
            broken_image += 1
            print(f"[이미지 오류] {image_path.name}: {e}")
            continue

        if width <= 0 or height <= 0:
            broken_image += 1
            continue

        # JSON metadata와 실제 이미지 크기 비교
        description = data.get("description", {})

        json_width = description.get("imageWidth")
        json_height = description.get("imageHeight")

        if (
            json_width
            and json_height
            and (
                json_width != width
                or json_height != height
            )
        ):
            metadata_size_mismatch += 1

        # =========================
        # PM annotation
        # =========================

        pm_objects = (
            data
            .get("annotations", {})
            .get("PM", [])
        )

        labels = []

        for obj in pm_objects:

            code = str(obj.get("PM_code", ""))

            # 킥보드만 사용
            if code not in KICKBOARD_CODES:
                continue

            # bbox만 사용
            if obj.get("shape_type") != "bbox":
                continue

            points = obj.get("points")

            if (
                not points
                or not isinstance(points, (list, tuple))
                or len(points) != 4
            ):
                invalid_points += 1
                continue

            try:
                x, y, w, h = map(float, points)

            except (TypeError, ValueError):
                invalid_points += 1
                continue

            # 잘못된 bbox 크기
            if w <= 0 or h <= 0:
                invalid_size += 1
                continue

            original_x1 = x
            original_y1 = y
            original_x2 = x + w
            original_y2 = y + h

            # 실제 이미지와 완전히 겹치지 않는 bbox 제외
            if (
                original_x1 >= width
                or original_y1 >= height
                or original_x2 <= 0
                or original_y2 <= 0
            ):
                fully_outside_boxes += 1
                continue

            # 이미지 경계에 걸친 bbox clipping
            x1 = max(0.0, original_x1)
            y1 = max(0.0, original_y1)
            x2 = min(float(width), original_x2)
            y2 = min(float(height), original_y2)

            if x2 <= x1 or y2 <= y1:
                fully_outside_boxes += 1
                continue

            if (
                x1 != original_x1
                or y1 != original_y1
                or x2 != original_x2
                or y2 != original_y2
            ):
                clipped_boxes += 1

            clipped_w = x2 - x1
            clipped_h = y2 - y1

            # YOLO 좌표 변환
            xc, yc, nw, nh = convert_bbox(
                x1,
                y1,
                clipped_w,
                clipped_h,
                width,
                height
            )

            # 최종 안전 검사
            if not (
                0.0 <= xc <= 1.0
                and 0.0 <= yc <= 1.0
                and 0.0 < nw <= 1.0
                and 0.0 < nh <= 1.0
            ):
                invalid_points += 1
                continue

            # YOLO class 0 = kickboard
            labels.append(
                f"0 {xc:.6f} {yc:.6f} "
                f"{nw:.6f} {nh:.6f}"
            )

            code_counter[code] += 1
            total_boxes += 1

        # 킥보드가 없는 이미지는 학습 데이터에서 제외
        if not labels:
            continue

        # =========================
        # clip 그룹 생성
        # =========================

        info = data.get("info", {})

        video_id = str(
            info.get("video_id", "unknown")
        )

        clip_id = str(
            info.get("clip_id", "unknown")
        )

        group_id = f"{video_id}_{clip_id}"

        groups[group_id].append(
            (image_path, labels)
        )

        if idx % 500 == 0:
            print(
                f"{idx}/{len(images)} "
                f"이미지 확인 완료"
            )

    # =========================
    # 추출 결과
    # =========================

    total_images = sum(
        len(items)
        for items in groups.values()
    )

    print()
    print("===== 추출 결과 =====")
    print(f"전체 원본 이미지       : {len(images)}")
    print(f"킥보드 이미지          : {total_images}")
    print(f"킥보드 bbox            : {total_boxes}")
    print(f"clip 그룹              : {len(groups)}")

    print()
    print("===== 데이터 검사 =====")
    print(f"JSON 없는 이미지       : {missing_json}")
    print(f"JSON 읽기 실패         : {broken_json}")
    print(f"이미지 읽기 실패       : {broken_image}")
    print(f"실제/JSON 해상도 불일치: {metadata_size_mismatch}")
    print(f"points 오류            : {invalid_points}")
    print(f"bbox 크기 오류         : {invalid_size}")
    print(f"완전히 화면 밖 bbox    : {fully_outside_boxes}")
    print(f"경계에서 잘린 bbox     : {clipped_boxes}")

    print()
    print("===== PM_code =====")

    for code in sorted(KICKBOARD_CODES):
        print(
            f"PM_code {code}: "
            f"{code_counter[code]}"
        )

    if not groups:
        print("킥보드 학습 데이터를 찾지 못했습니다.")
        return

    # =========================
    # Train / Val split
    # =========================
    # 같은 clip은 train/val로 나누지 않음.
    # 이미지 수 기준 약 80:20.

    target_val_images = round(
        total_images * VAL_RATIO
    )

    group_ids = list(groups.keys())

    # 같은 크기의 그룹에 대해 seed 기반 순서 사용
    random.shuffle(group_ids)

    # 큰 clip부터 확인
    group_ids.sort(
        key=lambda gid: len(groups[gid]),
        reverse=True
    )

    val_groups = set()
    val_images = 0

    for group_id in group_ids:

        group_size = len(groups[group_id])

        current_difference = abs(
            target_val_images - val_images
        )

        new_difference = abs(
            target_val_images
            - (val_images + group_size)
        )

        # 이 clip을 추가했을 때
        # 목표 20%에 더 가까워지는 경우만 추가
        if new_difference <= current_difference:
            val_groups.add(group_id)
            val_images += group_size

    train_groups = (
        set(group_ids) - val_groups
    )

    # 안전장치
    if not val_groups:

        smallest_group = min(
            group_ids,
            key=lambda gid: len(groups[gid])
        )

        val_groups.add(smallest_group)
        train_groups.discard(smallest_group)

    if not train_groups:

        largest_val_group = max(
            val_groups,
            key=lambda gid: len(groups[gid])
        )

        val_groups.remove(largest_val_group)
        train_groups.add(largest_val_group)

    # =========================
    # 출력 폴더 초기화
    # =========================

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val"]:

        (
            OUTPUT_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    split_stats = {
        "train_images": 0,
        "train_boxes": 0,
        "val_images": 0,
        "val_boxes": 0,
    }

    # =========================
    # 이미지 / 라벨 복사
    # =========================

    for group_id, items in groups.items():

        split = (
            "val"
            if group_id in val_groups
            else "train"
        )

        for image_path, labels in items:

            destination_image = (
                OUTPUT_ROOT
                / "images"
                / split
                / image_path.name
            )

            destination_label = (
                OUTPUT_ROOT
                / "labels"
                / split
                / f"{image_path.stem}.txt"
            )

            shutil.copy2(
                image_path,
                destination_image
            )

            destination_label.write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8"
            )

            split_stats[f"{split}_images"] += 1
            split_stats[f"{split}_boxes"] += len(labels)

    # =========================
    # data.yaml
    # =========================

    yaml_content = """path: /content/kickboard_cctv2

train: images/train
val: images/val

names:
  0: kickboard
"""

    (
        OUTPUT_ROOT
        / "data.yaml"
    ).write_text(
        yaml_content,
        encoding="utf-8"
    )

    # =========================
    # 최종 통계
    # =========================

    print()
    print("===== Dataset Split =====")

    print(
        f"목표 VAL 이미지        : "
        f"{target_val_images}"
    )

    print(
        f"TRAIN                  : "
        f"{split_stats['train_images']} images / "
        f"{split_stats['train_boxes']} boxes"
    )

    print(
        f"VAL                    : "
        f"{split_stats['val_images']} images / "
        f"{split_stats['val_boxes']} boxes"
    )

    print(
        f"Train clip groups      : "
        f"{len(train_groups)}"
    )

    print(
        f"Val clip groups        : "
        f"{len(val_groups)}"
    )

    print()
    print(
        f"생성 위치:"
        f"\n{OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()