from pathlib import Path
from collections import Counter
import hashlib


DATASET = Path(r".\data\AquaYOLO_26class_balanced")

SPLITS = ["train", "val", "test"]


CLASS_NAMES = [
    "jellyfish",
    "penguin",
    "puffin",
    "starfish",
    "stingray",
    "basking_shark",
    "blacktip_reef_shark",
    "great_white_shark",
    "hammerhead_shark",
    "nurse_shark",
    "oceanic_whitetip_shark",
    "thresher_shark",
    "whale_shark",
    "whitetip_reef_shark",
    "ballan_wrasse",
    "cuckoo_wrasse",
    "corkwing_wrasse",
    "goldsinny_wrasse",
    "rock_cook",
    "atlantic_cod",
    "pollack",
    "whiting",
    "spiny_dogfish",
    "brown_crab",
    "two_spot_goby",
    "human",
]


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def find_images(folder):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def label_path_for_image(image_path):
    return (
        image_path.parent.parent
        / "labels"
        / image_path.with_suffix(".txt").name
    )


def audit_split(split):
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = find_images(image_dir)

    class_counts = Counter()
    box_count = 0
    empty_labels = 0
    missing_labels = 0
    invalid_lines = 0

    width_stats = []
    height_stats = []

    images_with_multiple_objects = 0

    for image_path in images:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            missing_labels += 1
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if len(lines) == 0:
            empty_labels += 1
            continue

        image_objects = 0

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                invalid_lines += 1
                continue

            if not (
                0 <= class_id < len(CLASS_NAMES)
                and 0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_lines += 1
                continue

            class_counts[class_id] += 1
            box_count += 1
            image_objects += 1

            width_stats.append(w)
            height_stats.append(h)

        if image_objects >= 2:
            images_with_multiple_objects += 1

    return {
        "images": len(images),
        "boxes": box_count,
        "class_counts": class_counts,
        "empty_labels": empty_labels,
        "missing_labels": missing_labels,
        "invalid_lines": invalid_lines,
        "multi_object_images": images_with_multiple_objects,
        "width_stats": width_stats,
        "height_stats": height_stats,
    }


print("=" * 80)
print("AquaYOLO DATASET AUDIT")
print("=" * 80)
print()
print("Dataset:")
print(DATASET.resolve())
print()

total_images = 0
total_boxes = 0
total_empty = 0
total_missing = 0
total_invalid = 0

global_counts = Counter()


for split in SPLITS:

    result = audit_split(split)

    print("=" * 80)
    print(f"{split.upper()}")
    print("=" * 80)

    print(f"Images                 : {result['images']}")
    print(f"Annotated boxes        : {result['boxes']}")
    print(f"Empty label files      : {result['empty_labels']}")
    print(f"Missing label files    : {result['missing_labels']}")
    print(f"Invalid annotation rows: {result['invalid_lines']}")
    print(
        f"Images with 2+ objects : "
        f"{result['multi_object_images']}"
    )

    print()
    print("Class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = result["class_counts"].get(
            class_id,
            0,
        )

        print(
            f"{class_id:2d}  "
            f"{class_name:25s} "
            f"{count:6d}"
        )

    print()

    total_images += result["images"]
    total_boxes += result["boxes"]
    total_empty += result["empty_labels"]
    total_missing += result["missing_labels"]
    total_invalid += result["invalid_lines"]

    global_counts.update(
        result["class_counts"]
    )


print("=" * 80)
print("TOTAL")
print("=" * 80)

print(f"Images                  : {total_images}")
print(f"Annotated boxes         : {total_boxes}")
print(f"Empty label files       : {total_empty}")
print(f"Missing label files     : {total_missing}")
print(f"Invalid annotation rows : {total_invalid}")

print()
print("GLOBAL CLASS COUNTS")
print("=" * 80)

for class_id, class_name in enumerate(CLASS_NAMES):

    count = global_counts.get(
        class_id,
        0,
    )

    print(
        f"{class_id:2d}  "
        f"{class_name:25s} "
        f"{count:6d}"
    )

print()

zero_classes = [
    CLASS_NAMES[i]
    for i in range(len(CLASS_NAMES))
    if global_counts.get(i, 0) == 0
]

print("Classes with ZERO annotations:")
print(", ".join(zero_classes) if zero_classes else "None")

print()

print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)