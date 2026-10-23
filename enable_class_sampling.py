"""Safely patch AquaYOLO's train.py to use bounded class-aware image sampling.

Run from the AquaYOLO project root:
    python enable_class_sampling.py

The script backs up src/aquayolo/train.py and modifies only the training sampler.
It does not train, edit model architecture, or overwrite checkpoints.
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

TRAIN = Path("src/aquayolo/train.py")
MARKER = "# AQUAYOLO_CLASS_AWARE_SAMPLER_HELPER"
CLASS_MARKER = "class AquaYOLODataset(Dataset):"
LOADER_PATTERN = re.compile(r"(?m)^(?P<indent>[ \t]*)train_loader\s*=\s*DataLoader\s*\(")

HELPER = r'''# AQUAYOLO_CLASS_AWARE_SAMPLER_HELPER

def build_class_aware_sampler(data_root, dataset):
    """Build a bounded image-level sampler from YOLO train labels.

    Images containing rare classes receive higher sampling probability.
    Weights are based on inverse square-root class frequency, normalized,
    and clipped to avoid extreme oversampling. Empty-label images are kept
    in the training pool with a lower, non-zero sampling weight.
    """
    from pathlib import Path
    import math
    import statistics
    import torch
    from torch.utils.data import WeightedRandomSampler

    root = Path(data_root)
    label_dir = root / "labels" / "train"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Training label folder not found: {label_dir}")

    # Prefer the dataset's own image ordering if it exposes a conventional path list.
    image_paths = None
    for attr in ("image_paths", "image_files", "img_files", "images", "files", "paths"):
        value = getattr(dataset, attr, None)
        if isinstance(value, (list, tuple)) and len(value) == len(dataset):
            try:
                candidate = [Path(item) for item in value]
                if all(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} for p in candidate):
                    image_paths = candidate
                    break
            except (TypeError, ValueError):
                pass

    if image_paths is not None:
        label_paths = [label_dir / (p.stem + ".txt") for p in image_paths]
        missing = [str(p) for p in label_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Some training images have no corresponding label file; "
                f"for example: {missing[0]}"
            )
    else:
        # AquaYOLO's dataset uses a sorted image list; sorted YOLO label names
        # align with it when all image/label files are present.
        label_paths = sorted(label_dir.glob("*.txt"))
        if len(label_paths) != len(dataset):
            raise RuntimeError(
                "Cannot safely align label files with the dataset: "
                f"{len(label_paths)} train labels vs {len(dataset)} dataset items. "
                "No sampler was applied."
            )

    if len(label_paths) != len(dataset):
        raise RuntimeError(
            f"Sampler alignment error: {len(label_paths)} labels vs {len(dataset)} images."
        )

    try:
        num_classes = int(NUM_CLASSES)
    except Exception as exc:
        raise RuntimeError("NUM_CLASSES is unavailable while building sampler") from exc

    present_classes = []
    counts = [0] * num_classes
    for label_path in label_paths:
        classes_here = set()
        with label_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Bad class id in {label_path}:{line_no}") from exc
                if cls_id < 0 or cls_id >= num_classes:
                    raise ValueError(
                        f"Class id {cls_id} out of range in {label_path}:{line_no}"
                    )
                classes_here.add(cls_id)
                counts[cls_id] += 1
        present_classes.append(classes_here)

    positive_counts = [n for n in counts if n > 0]
    if not positive_counts:
        raise RuntimeError("No valid class annotations found in train labels.")
    median_freq = float(statistics.median(positive_counts))

    raw_weights = []
    for classes_here in present_classes:
        if not classes_here:
            raw_weights.append(0.5)
            continue
        # Inverse square-root frequency is intentionally less aggressive than
        # inverse frequency. Average across classes present in each image.
        rarity = [math.sqrt(median_freq / counts[c]) for c in classes_here if counts[c] > 0]
        raw_weights.append(sum(rarity) / max(len(rarity), 1))

    mean_weight = sum(raw_weights) / len(raw_weights)
    # Bound relative sampling probability: no image gets >3x the normalized
    # average weight; empty images retain a non-zero floor for negatives.
    weights = [min(3.0, max(0.25, w / max(mean_weight, 1e-12))) for w in raw_weights]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )

    print("Class-aware sampler enabled:")
    print(f"  Training images: {len(dataset)}")
    print(f"  Annotated class counts: {counts}")
    print(f"  Image sampling weight range: {min(weights):.3f} to {max(weights):.3f}")
    print("  Sampling: bounded inverse-square-root class frequency, with replacement")
    return sampler


'''


def main() -> int:
    if not TRAIN.is_file():
        print(f"ERROR: {TRAIN} was not found. Run this script from AquaYOLO project root.")
        return 2

    source = TRAIN.read_text(encoding="utf-8")
    if MARKER in source:
        print("Class-aware sampler helper already exists. No changes made.")
        if "sampler=train_sampler" in source:
            print("Training DataLoader already appears to use train_sampler.")
        return 0

    if CLASS_MARKER not in source:
        print(f"ERROR: Could not find expected insertion point: {CLASS_MARKER}")
        print("No changes made.")
        return 2

    loader_match = LOADER_PATTERN.search(source)
    if not loader_match:
        print("ERROR: Could not find 'train_loader = DataLoader('. No changes made.")
        return 2

    loader_start = loader_match.start()
    # Locate shuffle=True within this specific DataLoader argument block, before
    # the next top-level loader assignment or an adequately bounded text window.
    next_loader = re.search(r"(?m)^\s{0,8}\w+_loader\s*=\s*DataLoader\s*\(", source[loader_match.end():])
    block_end = loader_match.end() + next_loader.start() if next_loader else min(len(source), loader_match.end() + 2000)
    loader_block = source[loader_start:block_end]
    shuffle_match = re.search(r"\bshuffle\s*=\s*True\b", loader_block)
    if not shuffle_match:
        print("ERROR: Found train_loader but did not find shuffle=True in its arguments.")
        print("No changes made; manual inspection is needed to avoid changing the wrong loader.")
        return 2

    # Keep a timestamped source backup before editing.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TRAIN.with_name(f"train.py.bak_class_sampling_{stamp}")
    shutil.copy2(TRAIN, backup)

    patched = source.replace(CLASS_MARKER, HELPER + CLASS_MARKER, 1)
    # Add isolated import; avoids assumptions about the existing torch imports.
    import_anchor = "import torch\n"
    if "from torch.utils.data import WeightedRandomSampler" not in patched:
        if import_anchor not in patched:
            print("ERROR: Could not find 'import torch' to anchor the import. Restoring backup.")
            shutil.copy2(backup, TRAIN)
            return 2
        patched = patched.replace(import_anchor, import_anchor + "from torch.utils.data import WeightedRandomSampler\n", 1)

    # Insert sampler construction directly before the training DataLoader.
    # Use data root and the already constructed train_dataset.
    loader_match2 = LOADER_PATTERN.search(patched)
    if not loader_match2:
        print("ERROR: The training loader disappeared after insertion. Restoring backup.")
        shutil.copy2(backup, TRAIN)
        return 2
    indent = loader_match2.group("indent")
    sampler_line = f"{indent}train_sampler = build_class_aware_sampler(args.data, train_dataset)\n"
    patched = patched[:loader_match2.start()] + sampler_line + patched[loader_match2.start():]

    # Replace shuffle=True only inside the train_loader block.
    loader_match3 = LOADER_PATTERN.search(patched)
    assert loader_match3 is not None
    next_loader = re.search(r"(?m)^\s{0,8}\w+_loader\s*=\s*DataLoader\s*\(", patched[loader_match3.end():])
    block_end = loader_match3.end() + next_loader.start() if next_loader else min(len(patched), loader_match3.end() + 2000)
    loader_block = patched[loader_match3.start():block_end]
    shuffle_match = re.search(r"\bshuffle\s*=\s*True\b", loader_block)
    if not shuffle_match:
        print("ERROR: Could not safely patch shuffle setting. Restoring backup.")
        shutil.copy2(backup, TRAIN)
        return 2
    absolute_start = loader_match3.start() + shuffle_match.start()
    absolute_end = loader_match3.start() + shuffle_match.end()
    patched = patched[:absolute_start] + "shuffle=False, sampler=train_sampler" + patched[absolute_end:]

    TRAIN.write_text(patched, encoding="utf-8")

    # Lightweight structural verification. Full syntax check is done by launching
    # this same Python interpreter to compile the modified source.
    try:
        compile(patched, str(TRAIN), "exec")
    except SyntaxError as exc:
        shutil.copy2(backup, TRAIN)
        print(f"ERROR: Patched file failed syntax compilation: {exc}. Original restored.")
        return 2

    print("SUCCESS: class-aware sampling is now enabled in src/aquayolo/train.py")
    print(f"Backup: {backup}")
    print("Only training image sampling was changed; model architecture and loss were not altered.")
    print("Next: run a short separate experiment and evaluate on validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
