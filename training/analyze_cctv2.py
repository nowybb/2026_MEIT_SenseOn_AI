from pathlib import Path
import json
from collections import Counter

IMAGE_ROOT = Path("/Users/yeonwoo/Desktop/CCTV2")
LABEL_ROOT = Path("/Users/yeonwoo/Desktop/CCTV2_labeled")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# 샘플에서 직접 확인한 kickboard PM_code
KICKBOARD_CODES = {"29", "30", "35", "36"}


def main():
    print("이미지 목록 읽는 중...")

    images = [
        p for p in IMAGE_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    print("JSON 목록 읽는 중...")

    json_files = list(LABEL_ROOT.rglob("*.json"))

    # stem 기준 JSON 검색용 인덱스
    json_map = {}

    duplicate_json_stems = 0

    for p in json_files:
        if p.stem in json_map:
            duplicate_json_stems += 1
        else:
            json_map[p.stem] = p

    matched = 0
    missing_json = 0

    kickboard_images = 0
    kickboard_boxes = 0

    pm_counter = Counter()
    kickboard_counter = Counter()

    kickboard_examples = []

    for idx, image_path in enumerate(images, 1):

        json_path = json_map.get(image_path.stem)

        if json_path is None:
            missing_json += 1
            continue

        matched += 1

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[JSON ERROR] {json_path}: {e}")
            continue

        annotations = data.get("annotations", {})
        pm_objects = annotations.get("PM", [])

        image_kickboard_count = 0

        for obj in pm_objects:

            code = str(obj.get("PM_code", ""))

            pm_counter[code] += 1

            if code in KICKBOARD_CODES:
                kickboard_counter[code] += 1
                image_kickboard_count += 1

        if image_kickboard_count > 0:

            kickboard_images += 1
            kickboard_boxes += image_kickboard_count

            if len(kickboard_examples) < 10:
                kickboard_examples.append(
                    (
                        image_path.name,
                        json_path.name,
                        image_kickboard_count
                    )
                )

        if idx % 500 == 0:
            print(f"{idx}/{len(images)} 이미지 확인 완료")

    print()
    print("===== CCTV2 분석 결과 =====")
    print(f"전체 이미지              : {len(images)}")
    print(f"전체 JSON                : {len(json_files)}")
    print(f"이미지-JSON 매칭         : {matched}")
    print(f"JSON 없는 이미지         : {missing_json}")
    print(f"중복 JSON stem           : {duplicate_json_stems}")

    print()
    print("===== 킥보드 =====")
    print(f"킥보드 포함 이미지       : {kickboard_images}")
    print(f"킥보드 bbox 총 개수      : {kickboard_boxes}")

    print()
    print("===== 매칭 이미지의 전체 PM_code 분포 =====")

    for code, count in sorted(
        pm_counter.items(),
        key=lambda x: int(x[0]) if x[0].isdigit() else 999999
    ):
        print(f"PM_code {code}: {count}")

    print()
    print("===== 킥보드 PM_code 분포 =====")

    for code, count in sorted(
        kickboard_counter.items(),
        key=lambda x: int(x[0])
    ):
        print(f"PM_code {code}: {count}")

    print()
    print("===== 킥보드 이미지 예시 =====")

    for image_name, json_name, count in kickboard_examples:
        print(
            f"{image_name}  <->  {json_name} "
            f"(kickboard bbox: {count})"
        )


if __name__ == "__main__":
    main()