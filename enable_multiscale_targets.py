from datetime import datetime
from pathlib import Path
import py_compile
import shutil
import sys

ROOT = Path.cwd()
LOSS_PATH = ROOT / "src" / "aquayolo" / "loss.py"

if not LOSS_PATH.is_file():
    raise SystemExit(
        "ERROR: src/aquayolo/loss.py was not found. "
        "Run this script from the AquaYOLO project root. No files changed."
    )

source = LOSS_PATH.read_text(encoding="utf-8")
start = source.find("def build_targets(")
end_marker = "\n# ============================================================\n# CLASS WEIGHTS"
end = source.find(end_marker, start + 1) if start >= 0 else -1

if start < 0 or end < 0:
    raise SystemExit(
        "ERROR: expected build_targets()/CLASS WEIGHTS boundary was not found. "
        "No files changed."
    )

old_func = source[start:end]
if "MULTI_SCALE_TARGET_ASSIGNMENT_EXPERIMENT" in old_func:
    raise SystemExit(
        "This multi-scale target-assignment experiment is already enabled. "
        "No files changed."
    )

new_func = '''def build_targets(
    targets,
    batch_size,
    nc,
    grid_sizes,
):
    """Create targets on the primary detection scale and one adjacent scale.

    This experiment keeps the model architecture unchanged. Each object is
    assigned to its existing size-based scale plus one adjacent scale to test
    whether cross-scale supervision improves recall/generalization. If two
    objects land on the same cell at a scale, this implementation follows the
    existing one-target-per-cell behavior (the later object overwrites it).
    """
    if len(targets) > 0:
        device = targets[0].device
    else:
        device = torch.device("cpu")

    results = []
    for grid_h, grid_w in grid_sizes:
        obj_target = torch.zeros(
            batch_size, 1, grid_h, grid_w, device=device
        )
        cls_target = torch.full(
            (batch_size, grid_h, grid_w),
            -1,
            dtype=torch.long,
            device=device,
        )
        box_target = torch.zeros(
            batch_size, 4, grid_h, grid_w, device=device
        )
        box_mask = torch.zeros(
            batch_size, 1, grid_h, grid_w, device=device
        )
        results.append((obj_target, cls_target, box_target, box_mask))

    for batch_index in range(batch_size):
        if batch_index >= len(targets):
            continue

        current_targets = targets[batch_index]
        if current_targets is None or current_targets.numel() == 0:
            continue

        for row in current_targets:
            if row.numel() != 5:
                continue

            class_id = int(row[0].item())
            if class_id < 0 or class_id >= nc:
                continue

            cx = float(row[1].clamp(0.0, 1.0).item())
            cy = float(row[2].clamp(0.0, 1.0).item())
            w = float(row[3].clamp(0.0, 1.0).item())
            h = float(row[4].clamp(0.0, 1.0).item())
            if w <= 0.0 or h <= 0.0:
                continue

            primary_scale = choose_detection_scale(w, h)

            # MULTI_SCALE_TARGET_ASSIGNMENT_EXPERIMENT
            # Assign the primary scale plus its closest adjacent scale.
            if primary_scale == 0:
                scales_to_assign = (0, 1)
            elif primary_scale == 2:
                scales_to_assign = (1, 2)
            else:
                # For the middle band, select the adjacent scale based on
                # object size, using the same 640-pixel reference as the
                # current scale thresholds (64 and 192 pixels).
                size_pixels = max(w, h) * 640.0
                secondary_scale = 0 if size_pixels < 128.0 else 2
                scales_to_assign = (primary_scale, secondary_scale)

            for scale in sorted(set(scales_to_assign)):
                grid_h, grid_w = grid_sizes[scale]
                gx = cx * grid_w
                gy = cy * grid_h
                gi = min(max(int(gx), 0), grid_w - 1)
                gj = min(max(int(gy), 0), grid_h - 1)

                obj_target, cls_target, box_target, box_mask = results[scale]
                obj_target[batch_index, 0, gj, gi] = 1.0
                cls_target[batch_index, gj, gi] = class_id
                box_target[batch_index, :, gj, gi] = torch.tensor(
                    [cx, cy, w, h], dtype=torch.float32, device=device
                )
                box_mask[batch_index, 0, gj, gi] = 1.0

    return results
'''

backup = LOSS_PATH.with_name(
    f"loss.py.backup_multiscale_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
shutil.copy2(LOSS_PATH, backup)
updated = source[:start] + new_func.rstrip() + source[end:]
LOSS_PATH.write_text(updated, encoding="utf-8")

try:
    py_compile.compile(str(LOSS_PATH), doraise=True)
except Exception as exc:
    shutil.copy2(backup, LOSS_PATH)
    raise SystemExit(
        f"ERROR: patched loss.py failed syntax check and was restored from {backup.name}: {exc}"
    )

print("SUCCESS: multi-scale target-assignment experiment enabled.")
print(f"Backup: {backup}")
print("Changed only build_targets() in src/aquayolo/loss.py.")
print("Python syntax check passed.")
print("Existing checkpoints and dataset files were not modified.")
print("This is an experiment; evaluate validation mAP before keeping it.")
