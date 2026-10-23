"""Prepare a controlled AquaYOLO false-positive-reduction experiment.

Run from the AquaYOLO project root:
    python prepare_fp_reduction_experiment.py

Changes only:
1. Removes the experimental training-only augmentation block, restoring the
   deterministic training preprocessing used by the earlier best checkpoint.
2. Raises the hard-negative objectness term multiplier from 1.0 to 1.5.

Backs up both files first; rolls both back if compilation fails.
"""
from datetime import datetime
from pathlib import Path
import py_compile
import shutil
import sys

TRAIN = Path("src/aquayolo/train.py")
LOSS = Path("src/aquayolo/loss.py")
AUG_MARKER = "        # AQUAYOLO_TRAIN_AUGMENTATION_V1\n"
RETURN_LINE = "        return image_tensor, target"


def main() -> int:
    if not TRAIN.is_file() or not LOSS.is_file():
        print("ERROR: run from project root; src/aquayolo/train.py and loss.py must exist.")
        return 2

    train_src = TRAIN.read_text(encoding="utf-8")
    loss_src = LOSS.read_text(encoding="utf-8")

    # Remove only our known augmentation block, if it is present.
    train_updated = train_src
    if AUG_MARKER in train_src:
        start = train_src.index(AUG_MARKER)
        end = train_src.find(RETURN_LINE, start)
        if end < 0:
            print("STOP: found augmentation marker but not its expected return line; no changes made.")
            return 2
        end += len(RETURN_LINE)
        train_updated = train_src[:start] + RETURN_LINE + train_src[end:]
        print("Will revert the experimental image augmentation block.")
    else:
        print("No experimental augmentation block found; train.py will remain unchanged.")

    # Change the multiplier only inside balanced_objectness_loss.
    func_start = loss_src.find("def balanced_objectness_loss(")
    func_end = loss_src.find("# CLASSIFICATION LOSS", func_start + 1)
    if func_start < 0 or func_end < 0:
        print("STOP: could not identify balanced_objectness_loss boundaries; no changes made.")
        return 2
    func = loss_src[func_start:func_end]
    old = "1.0 * hard_negative_loss"
    new = "1.5 * hard_negative_loss"
    if func.count(old) != 1:
        print("STOP: expected exactly one hard-negative multiplier of 1.0; no changes made.")
        print("This prevents accidentally patching an unexpected loss version.")
        return 2
    loss_updated = loss_src[:func_start] + func.replace(old, new, 1) + loss_src[func_end:]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_backup = TRAIN.with_name(f"train.py.backup_before_fp_experiment_{stamp}.bak")
    loss_backup = LOSS.with_name(f"loss.py.backup_before_fp_experiment_{stamp}.bak")
    shutil.copy2(TRAIN, train_backup)
    shutil.copy2(LOSS, loss_backup)

    try:
        TRAIN.write_text(train_updated, encoding="utf-8")
        LOSS.write_text(loss_updated, encoding="utf-8")
        py_compile.compile(str(TRAIN), doraise=True)
        py_compile.compile(str(LOSS), doraise=True)
    except Exception as exc:
        shutil.copy2(train_backup, TRAIN)
        shutil.copy2(loss_backup, LOSS)
        print("ERROR: syntax check failed; both source files restored from backups.")
        print(exc)
        return 2

    print("SUCCESS: prepared the controlled false-positive-reduction experiment.")
    print(f"train.py backup: {train_backup}")
    print(f"loss.py backup : {loss_backup}")
    print("Training-only augmentation: reverted if it was enabled.")
    print("Hard-negative objectness multiplier: 1.0 -> 1.5.")
    print("AquaYOLO architecture, dataset files, and existing checkpoints unchanged.")
    print("Python syntax checks passed for both files.")
    print("This is an experiment; evaluate validation mAP before keeping it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
