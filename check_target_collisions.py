from pathlib import Path
from collections import defaultdict

DATASET = Path(r"data\AquaYOLO_26class_balanced")
IMAGE_SIZE = 640

SPLITS = ["train", "val", "test"]
GRID_SIZES = [
    (80, 80),   # P3
    (40, 40),   # P4
    (20, 20),   # P5
]


def choose_scale(w, h):
    size_pixels = ((w * h) ** 0.5) * IMAGE_SIZE

    if size_pixels < 64.0:
        return 0

    if size_pixels < 192.0:
        return 1

    return 2


for split in SPLITS:
    label_dir = DATASET / "labels" / split

    total_objects = 0
    images_with_collision = 0
    total_collisions = 0

    for label_file in label_dir.glob("*.txt"):
        occupied = defaultdict(list)

        lines = label_file.read_text(encoding="utf-8").splitlines()

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                continue

            cls_id, cx, cy, w, h = map(float, parts)

            if w <= 0 or h <= 0:
                continue

            total_objects += 1

            scale = choose_scale(w, h)
            grid_h, grid_w = GRID_SIZES[scale]

            gx = cx * grid_w
            gy = cy * grid_h

            gi = min(max(int(gx), 0), grid_w - 1)
            gj = min(max(int(gy), 0), grid_h - 1)

            key = (scale, gj, gi)
            occupied[key].append((line_no, int(cls_id)))

        collisions_here = 0

        for key, objects in occupied.items():
            if len(objects) > 1:
                collisions_here += len(objects) - 1

        if collisions_here > 0:
            images_with_collision += 1
            total_collisions += collisions_here

    print()
    print("=" * 70)
    print(split.upper())
    print("=" * 70)
    print("Objects                  :", total_objects)
    print("Images with collisions   :", images_with_collision)
    print("Overwritten targets      :", total_collisions)

    if total_objects > 0:
        print(
            "Collision rate           :"
            f" {100.0 * total_collisions / total_objects:.3f}%"
        )