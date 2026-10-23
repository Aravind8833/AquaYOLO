from pathlib import Path
import sys

import torch
from ultralytics import YOLO

sys.path.insert(0, "src")

from aquayolo.model import AquaYOLO


YOLO_WEIGHTS = Path("yolov8s.pt")


def tensor_shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return None


print("=" * 80)
print("AquaYOLO <-> YOLOv8s PRETRAINED COMPATIBILITY CHECK")
print("=" * 80)

# ------------------------------------------------------------
# AquaYOLO
# ------------------------------------------------------------
aqua = AquaYOLO(nc=26)
aqua_state = aqua.state_dict()

print(f"\nAquaYOLO parameters : {sum(p.numel() for p in aqua.parameters()):,}")
print(f"AquaYOLO state keys : {len(aqua_state)}")

# ------------------------------------------------------------
# YOLOv8s
# ------------------------------------------------------------
yolo = YOLO(str(YOLO_WEIGHTS))
yolo_state = yolo.model.state_dict()

print(f"YOLOv8s parameters  : {sum(p.numel() for p in yolo.model.parameters()):,}")
print(f"YOLOv8s state keys  : {len(yolo_state)}")

# ------------------------------------------------------------
# Exact shape matches
# ------------------------------------------------------------
aqua_by_shape = {}
for key, value in aqua_state.items():
    shape = tensor_shape(value)
    if shape is None:
        continue
    aqua_by_shape.setdefault(shape, []).append(key)

yolo_by_shape = {}
for key, value in yolo_state.items():
    shape = tensor_shape(value)
    if shape is None:
        continue
    yolo_by_shape.setdefault(shape, []).append(key)

exact_shape_pairs = []

for shape in sorted(set(aqua_by_shape) & set(yolo_by_shape)):
    a_keys = aqua_by_shape[shape]
    y_keys = yolo_by_shape[shape]

    exact_shape_pairs.append(
        (shape, len(a_keys), len(y_keys), a_keys, y_keys)
    )

print()
print("=" * 80)
print("EXACT SHAPE COMPATIBILITY")
print("=" * 80)

matched_aqua_tensors = 0
matched_yolo_tensors = 0

for shape, a_count, y_count, a_keys, y_keys in exact_shape_pairs:
    usable = min(a_count, y_count)

    matched_aqua_tensors += usable
    matched_yolo_tensors += usable

    print(
        f"\nShape {shape}"
        f"\n  AquaYOLO tensors : {a_count}"
        f"\n  YOLOv8s tensors  : {y_count}"
        f"\n  Possible pairs   : {usable}"
    )

    print("  Aqua examples:")
    for key in a_keys[:5]:
        print(f"    {key}")

    print("  YOLO examples:")
    for key in y_keys[:5]:
        print(f"    {key}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Possible exact-shape tensor pairs : {matched_aqua_tensors}")
print()

if matched_aqua_tensors == 0:
    print("No exact-shape tensors found.")
    print("Direct YOLOv8s weight transfer is not appropriate.")
else:
    print(
        "Exact-shape tensors exist, but this does NOT mean they are "
        "semantically corresponding layers."
    )
    print(
        "We must inspect the matches before transferring anything."
    )

print("=" * 80)