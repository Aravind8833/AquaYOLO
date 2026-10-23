
import sys
import torch

sys.path.insert(0, "src")
from aquayolo.model import AquaYOLO

aqua = AquaYOLO(nc=26).state_dict()

checkpoint = torch.load(
    "MegaFauna.pt",
    map_location="cpu",
    weights_only=False,
)
mega = checkpoint["model"].float().state_dict()

pairs = [
    ("model.0.", "backbone.stem."),
    ("model.1.", "backbone.p2_down."),
    ("model.2.", "backbone.p2_c2f."),
    ("model.3.", "backbone.p3_down."),
    ("model.4.", "backbone.p3_c2f."),
    ("model.5.", "backbone.p4_down."),
    ("model.6.", "backbone.p4_c2f."),
]

total = 0

print("MEGAFAUNA -> AQUAYOLO COMPATIBILITY")
for source, destination in pairs:
    matched = [
        key for key, tensor in mega.items()
        if key.startswith(source)
        and key.replace(source, destination, 1) in aqua
        and aqua[key.replace(source, destination, 1)].shape
        == tensor.shape
    ]
    print(f"{source} -> {destination}: {len(matched)} tensors")
    total += len(matched)

print("TOTAL COMPATIBLE TENSORS:", total)
print("Read-only check: no checkpoint was modified.")