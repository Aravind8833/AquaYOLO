
from pathlib import Path
import sys
import torch

sys.path.insert(0, "src")
from aquayolo.model import AquaYOLO

BEST = Path("runs/exp1_focal_5epoch/best.pt")
MEGA = Path("MegaFauna.pt")
OUT = Path("runs/exp1_megafauna_transfer/megafauna_init.pt")

if not BEST.is_file() or not MEGA.is_file():
    raise FileNotFoundError("Could not find best.pt or MegaFauna.pt.")

print("Loading current best AquaYOLO checkpoint...")
base_ckpt = torch.load(BEST, map_location="cpu", weights_only=False)
if not isinstance(base_ckpt, dict) or "model_state_dict" not in base_ckpt:
    raise ValueError("The current checkpoint has no model_state_dict.")

model = AquaYOLO(nc=26)
base_sd = {
    key: value.detach().clone()
    for key, value in base_ckpt["model_state_dict"].items()
}
# Strict validation: the current checkpoint must match this model exactly.
model.load_state_dict(base_sd, strict=True)

print("Loading MegaFauna checkpoint...")
mega_ckpt = torch.load(MEGA, map_location="cpu", weights_only=False)
mega_model = mega_ckpt["model"].float()
mega_sd = mega_model.state_dict()

# Only transfer complete modules with matching keys and tensor shapes.
# Deliberately skip C2f internal blocks, SPPF, neck and detection head.
mapping = [
    ("model.0.", "backbone.stem.", "Stem convolution"),
    ("model.1.", "backbone.p2_down.", "P2 downsample"),
    ("model.2.cv1.", "backbone.p2_c2f.cv1.", "P2 C2f input projection"),
    ("model.3.", "backbone.p3_down.", "P3 downsample"),
    ("model.4.cv1.", "backbone.p3_c2f.cv1.", "P3 C2f input projection"),
    ("model.5.", "backbone.p4_down.", "P4 downsample"),
    ("model.6.cv1.", "backbone.p4_c2f.cv1.", "P4 C2f input projection"),
]

transferred = []
failed = []

for source_prefix, target_prefix, description in mapping:
    source_keys = [
        key for key in mega_sd if key.startswith(source_prefix)
    ]
    target_keys = [
        key for key in base_sd if key.startswith(target_prefix)
    ]

    pairs = []
    for source_key in source_keys:
        target_key = target_prefix + source_key[len(source_prefix):]
        if target_key not in base_sd:
            continue
        if base_sd[target_key].shape != mega_sd[source_key].shape:
            continue
        pairs.append((source_key, target_key))

    # Do not partially replace a module.
    complete = (
        bool(source_keys)
        and len(pairs) == len(source_keys)
        and len(target_keys) == len(source_keys)
        and {target for _, target in pairs} == set(target_keys)
    )

    if not complete:
        failed.append(description)
        print(f"SKIPPED: {description} (not a complete exact match)")
        continue

    for source_key, target_key in pairs:
        base_sd[target_key] = mega_sd[source_key].detach().clone()
    transferred.extend(target for _, target in pairs)
    print(f"TRANSFERRED: {description} ({len(pairs)} tensors)")

if not transferred:
    raise RuntimeError("No complete modules matched. Nothing was saved.")

# Verify the assembled weights against the unchanged AquaYOLO architecture.
model.load_state_dict(base_sd, strict=True)

# Preserve the original best checkpoint; save a separate initialization.
out_ckpt = {
    key: value
    for key, value in base_ckpt.items()
    if key != "optimizer_state_dict"
}
out_ckpt["model_state_dict"] = base_sd
out_ckpt["transfer_note"] = (
    "Current AquaYOLO best plus complete, shape-matched MegaFauna modules"
)
out_ckpt["transferred_keys"] = transferred

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(out_ckpt, OUT)

# Reload and verify the saved file too.
saved = torch.load(OUT, map_location="cpu", weights_only=False)
AquaYOLO(nc=26).load_state_dict(saved["model_state_dict"], strict=True)

print("\nSUCCESS: initialization saved and strictly verified.")
print("Transferred tensors:", len(transferred))
print("Saved to:", OUT)
print("Original best checkpoint was not modified.")