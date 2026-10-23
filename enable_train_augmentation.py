"""Safely enable mild training-only image augmentation for AquaYOLO.

Run from the AquaYOLO project root:
    python enable_train_augmentation.py

Creates a timestamped backup before editing train.py. It also reverts the
previous experimental class-aware sampler to ordinary shuffled sampling so
this experiment tests augmentation without keeping the sampler change active.
"""
from __future__ import annotations

import py_compile
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

TRAIN = Path("src/aquayolo/train.py")
AUG_MARKER = "# AQUAYOLO_TRAIN_AUGMENTATION_V1"
SAMPLER_MARKER = "# AQUAYOLO_CLASS_AWARE_SAMPLER_HELPER"
DATASET_CLASS = "class AquaYOLODataset(Dataset):"
RETURN_ANCHOR = "        return image_tensor, target"

AUGMENTATION = '''        # AQUAYOLO_TRAIN_AUGMENTATION_V1
        # Training-only augmentation; validation and test stay deterministic.
        if self.split == "train":
            # Horizontal flip (YOLO target columns: class, cx, cy, w, h).
            if torch.rand(()) < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(2,))
                if target.numel() > 0:
                    target[:, 1] = 1.0 - target[:, 1]

            # Moderate brightness and contrast variation after the fixed
            # underwater enhancement/resize pipeline. Boxes do not change.
            if torch.rand(()) < 0.8:
                contrast = 0.9 + 0.2 * torch.rand(()).item()
                brightness = 0.9 + 0.2 * torch.rand(()).item()
                channel_mean = image_tensor.mean(dim=(1, 2), keepdim=True)
                image_tensor = (image_tensor - channel_mean) * contrast + channel_mean
                image_tensor = (image_tensor * brightness).clamp_(0.0, 1.0)

        return image_tensor, target'''


def main() -> int:
    if not TRAIN.is_file():
        print(f"ERROR: {TRAIN} not found. Run from the AquaYOLO project root.")
        return 2

    source = TRAIN.read_text(encoding="utf-8")
    if AUG_MARKER in source:
        print("STOP: augmentation is already enabled. No changes made.")
        return 2
    if source.count(RETURN_ANCHOR) != 1:
        print("STOP: expected one dataset return anchor; no changes made.")
        return 2

    patched = source
    sampler_removed = False

    # Remove the previous class-aware sampler helper, if installed.
    if SAMPLER_MARKER in patched:
        start = patched.index(SAMPLER_MARKER)
        end = patched.find(DATASET_CLASS, start)
        if end < 0:
            print("STOP: sampler helper end marker not found; no changes made.")
            return 2
        patched = patched[:start] + patched[end:]
        sampler_removed = True

    # Remove the sampler import if it was added by the sampler patch.
    patched = re.sub(
        r"(?m)^from torch\.utils\.data import WeightedRandomSampler\s*\n",
        "",
        patched,
    )

    # Remove sampler construction and restore ordinary shuffled sampling.
    sampler_line_pattern = re.compile(
        r"(?m)^[ \t]*train_sampler\s*=\s*build_class_aware_sampler\(args\.data,\s*train_dataset\)\s*\n"
    )
    if sampler_line_pattern.search(patched):
        patched = sampler_line_pattern.sub("", patched, count=1)
        sampler_removed = True

    loader_sampler_pattern = re.compile(
        r"shuffle\s*=\s*False\s*,\s*sampler\s*=\s*train_sampler"
    )
    if loader_sampler_pattern.search(patched):
        patched = loader_sampler_pattern.sub("shuffle=True", patched, count=1)
        sampler_removed = True
    elif sampler_removed:
        # If sampler construction existed but its DataLoader linkage did not,
        # don't risk saving a half-reverted experiment.
        if "train_sampler" in patched:
            print("STOP: class-aware sampler references remain; no changes made.")
            return 2

    patched = patched.replace(RETURN_ANCHOR, AUGMENTATION, 1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TRAIN.with_name(f"train.py.before_augmentation_{stamp}.bak")
    shutil.copy2(TRAIN, backup)
    TRAIN.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(TRAIN), doraise=True)
    except Exception as exc:
        shutil.copy2(backup, TRAIN)
        print(f"ERROR: syntax check failed; original restored from {backup}")
        print(exc)
        return 2

    print("SUCCESS: training-only augmentation enabled and syntax check passed.")
    print(f"Backup: {backup}")
    print("Augmentations: 50% horizontal flip with cx corrected; 80% mild brightness/contrast.")
    print("Validation/test preprocessing remains unchanged.")
    if sampler_removed:
        print("Class-aware sampler experiment reverted to ordinary shuffled sampling for a cleaner comparison.")
    else:
        print("No class-aware sampler modification was detected; ordinary loader configuration was retained.")
    print("Model architecture, loss, dataset files, and existing checkpoints were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
