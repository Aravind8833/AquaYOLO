from pathlib import Path
import shutil

SRC = Path(r"data\AquaYOLO_26class_final")
DST = Path(r"data\AquaYOLO_26class_balanced")

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(folder: Path, stem: str):
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


total_copied = 0

for split in ["train", "val", "test"]:
    src_labels = SRC / "labels" / split
    src_images = SRC / "images" / split

    dst_labels = DST / "labels" / split
    dst_images = DST / "images" / split

    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    for label_file in src_labels.glob("*.txt"):
        # Empty label = negative/background image
        if label_file.read_text(encoding="utf-8").strip():
            continue

        stem = label_file.stem
        image_file = find_image(src_images, stem)

        if image_file is None:
            print(f"WARNING: image not found for {label_file}")
            continue

        dst_image = dst_images / image_file.name
        dst_label = dst_labels / label_file.name

        # Avoid overwriting an existing labeled image
        if dst_image.exists() or dst_label.exists():
            continue

        shutil.copy2(image_file, dst_image)
        shutil.copy2(label_file, dst_label)

        total_copied += 1

    print(f"{split}: done")

print()
print("=" * 70)
print(f"Empty/background images restored: {total_copied}")
print("=" * 70)

if total_copied != 271:
    print("WARNING: Expected 271 images.")
else:
    print("SUCCESS: All 271 empty-label images restored.")