from pathlib import Path
import torch
from ultralytics import YOLO

import sys
sys.path.insert(0, "src")

from aquayolo.model import AquaYOLO


# ============================================================
# SETTINGS
# ============================================================

TEACHER_WEIGHTS = Path("FishInv.pt")

OUTPUT_DIR = Path("runs") / "pretrained_init_fishinv"
OUTPUT_PATH = OUTPUT_DIR / "aquayolo_fishinv_init.pt"

NC = 26


# ============================================================
# HELPERS
# ============================================================

def copy_tensor(
    source_state,
    target_state,
    source_key,
    target_key,
    transferred,
):
    """Copy one tensor only when shape matches."""

    if source_key not in source_state:
        print(f"[SKIP] Missing source: {source_key}")
        return False

    if target_key not in target_state:
        print(f"[SKIP] Missing target: {target_key}")
        return False

    source_tensor = source_state[source_key]
    target_tensor = target_state[target_key]

    if source_tensor.shape != target_tensor.shape:
        print(
            f"[SKIP] Shape mismatch:\n"
            f"       source: {source_key} {tuple(source_tensor.shape)}\n"
            f"       target: {target_key} {tuple(target_tensor.shape)}"
        )
        return False

    target_state[target_key].copy_(source_tensor)

    transferred.append(
        (
            source_key,
            target_key,
            tuple(source_tensor.shape),
        )
    )

    return True


def transfer_bn_module(
    source_state,
    target_state,
    source_prefix,
    target_prefix,
    transferred,
):
    """Transfer Conv+BN weights for compatible modules."""

    keys = [
        "bn.weight",
        "bn.bias",
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
    ]

    for suffix in keys:
        copy_tensor(
            source_state,
            target_state,
            f"{source_prefix}.{suffix}",
            f"{target_prefix}.{suffix}",
            transferred,
        )


def transfer_conv_bn(
    source_state,
    target_state,
    source_prefix,
    target_prefix,
    transferred,
):
    """Transfer conv.weight plus BatchNorm state."""

    copy_tensor(
        source_state,
        target_state,
        f"{source_prefix}.conv.weight",
        f"{target_prefix}.conv.weight",
        transferred,
    )

    transfer_bn_module(
        source_state,
        target_state,
        source_prefix,
        target_prefix,
        transferred,
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FISHINV -> AQUAYOLO PRETRAINED BACKBONE TRANSFER")
print("=" * 80)

if not TEACHER_WEIGHTS.exists():
    raise FileNotFoundError(
        f"FishInv checkpoint not found: {TEACHER_WEIGHTS.resolve()}"
    )

# ------------------------------------------------------------
# Load teacher
# ------------------------------------------------------------

print("\nLoading FishInv teacher...")

teacher = YOLO(str(TEACHER_WEIGHTS))
teacher_model = teacher.model
teacher_state = teacher_model.state_dict()

print(
    f"FishInv parameters: "
    f"{sum(p.numel() for p in teacher_model.parameters()):,}"
)

# ------------------------------------------------------------
# Create AquaYOLO
# ------------------------------------------------------------

print("\nCreating AquaYOLO...")

student = AquaYOLO(nc=NC)

student_state = student.state_dict()

print(
    f"AquaYOLO parameters: "
    f"{sum(p.numel() for p in student.parameters()):,}"
)

# ------------------------------------------------------------
# Compatible mappings
# ------------------------------------------------------------

transferred = []

print("\n" + "=" * 80)
print("TRANSFER")
print("=" * 80)

# ------------------------------------------------------------
# 1. Stem
#
# FishInv:
#   model.0
#
# AquaYOLO:
#   backbone.stem
# ------------------------------------------------------------

print("\n[1] Stem")

transfer_conv_bn(
    teacher_state,
    student_state,
    "model.0",
    "backbone.stem",
    transferred,
)


# ------------------------------------------------------------
# 2. P2 downsample
#
# FishInv:
#   model.1
#
# AquaYOLO:
#   backbone.p2_down
# ------------------------------------------------------------

print("\n[2] P2 downsample")

transfer_conv_bn(
    teacher_state,
    student_state,
    "model.1",
    "backbone.p2_down",
    transferred,
)


# ------------------------------------------------------------
# 3. P2 C2f first projection
#
# Both use:
#   128 -> 128
#
# We intentionally transfer ONLY cv1 because the internal
# Bottleneck structures are different.
# ------------------------------------------------------------

print("\n[3] P2 C2f cv1")

copy_tensor(
    teacher_state,
    student_state,
    "model.2.cv1.conv.weight",
    "backbone.p2_c2f.cv1.conv.weight",
    transferred,
)

transfer_bn_module(
    teacher_state,
    student_state,
    "model.2.cv1",
    "backbone.p2_c2f.cv1",
    transferred,
)


# ------------------------------------------------------------
# 4. P3 downsample
#
# FishInv:
#   model.3
#
# AquaYOLO:
#   backbone.p3_down
# ------------------------------------------------------------

print("\n[4] P3 downsample")

transfer_conv_bn(
    teacher_state,
    student_state,
    "model.3",
    "backbone.p3_down",
    transferred,
)


# ------------------------------------------------------------
# 5. P3 C2f first projection
#
# Both use:
#   256 -> 256
# ------------------------------------------------------------

print("\n[5] P3 C2f cv1")

copy_tensor(
    teacher_state,
    student_state,
    "model.4.cv1.conv.weight",
    "backbone.p3_c2f.cv1.conv.weight",
    transferred,
)

transfer_bn_module(
    teacher_state,
    student_state,
    "model.4.cv1",
    "backbone.p3_c2f.cv1",
    transferred,
)


# ------------------------------------------------------------
# 6. P4 downsample
#
# FishInv:
#   model.5
#
# AquaYOLO:
#   backbone.p4_down
# ------------------------------------------------------------

print("\n[6] P4 downsample")

transfer_conv_bn(
    teacher_state,
    student_state,
    "model.5",
    "backbone.p4_down",
    transferred,
)


# ------------------------------------------------------------
# 7. P4 C2f first projection
#
# Both use:
#   512 -> 512
# ------------------------------------------------------------

print("\n[7] P4 C2f cv1")

copy_tensor(
    teacher_state,
    student_state,
    "model.6.cv1.conv.weight",
    "backbone.p4_c2f.cv1.conv.weight",
    transferred,
)

transfer_bn_module(
    teacher_state,
    student_state,
    "model.6.cv1",
    "backbone.p4_c2f.cv1",
    transferred,
)


# ------------------------------------------------------------
# IMPORTANT:
# We intentionally DO NOT transfer:
#
# - FishInv layer 7
# - FishInv layer 8
# - FishInv SPPF
# - Detection head
#
# because AquaYOLO uses different dimensions/structure.
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("TRANSFER SUMMARY")
print("=" * 80)

print(
    f"Transferred tensors : {len(transferred)}"
)

if transferred:
    print("\nTransferred components:")

    for source_key, target_key, shape in transferred:
        print(
            f"  {source_key}"
            f"  ->  {target_key}"
            f"  shape={shape}"
        )

# ------------------------------------------------------------
# Load modified state into AquaYOLO
# ------------------------------------------------------------

student.load_state_dict(
    student_state,
    strict=True,
)

# ------------------------------------------------------------
# Save initialization checkpoint
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint = {
    "model_state_dict": student.state_dict(),
    "source": "FishInv.pt",
    "classes": NC,
    "transferred_tensors": transferred,
}

torch.save(
    checkpoint,
    OUTPUT_PATH,
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(
    f"Output: {OUTPUT_PATH.resolve()}"
)

print(
    "\nAqua-Attn, PANet, and the 26-class detection head "
    "remain AquaYOLO-specific and were NOT replaced."
)

print("=" * 80)