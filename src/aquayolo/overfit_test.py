import torch
from pathlib import Path

from aquayolo.model import AquaYOLO
from aquayolo.loss import aqua_loss
from aquayolo.train import AquaYOLODataset
from aquayolo.evaluate import decode_predictions
from aquayolo.loss import xywh_to_xyxy


# ============================================================
# SETTINGS
# ============================================================

DATASET = Path(
    r".\data\AquaYOLO_baseline_species26"
)

IMAGE_SIZE = 640
NC = 26

STEPS = 250
LR = 1e-3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("AquaYOLO OVERFIT SANITY TEST")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATASET
# ============================================================

dataset = AquaYOLODataset(
    root=DATASET,
    split="train",
    size=IMAGE_SIZE
)


# Find a training image with at least one object.
selected_index = None

for i in range(len(dataset)):

    image, target = dataset[i]

    if target.numel() > 0:
        selected_index = i
        break


if selected_index is None:
    raise RuntimeError(
        "Could not find a training image with annotations."
    )


image, target = dataset[
    selected_index
]

image = image.unsqueeze(0).to(
    device
)

target = [
    target.to(device)
]


print()
print(
    "Selected training image:",
    dataset.images[selected_index]
)

print(
    "Ground-truth objects:",
    len(target[0])
)

print(
    "Ground truth:",
    target[0].detach().cpu().tolist()
)


# ============================================================
# MODEL
# ============================================================

model = AquaYOLO(
    nc=NC
).to(device)

model.train()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0
)


# ============================================================
# TRAIN ON SAME IMAGE REPEATEDLY
# ============================================================

print()
print("=" * 70)
print("STARTING OVERFIT TEST")
print("=" * 70)

for step in range(
    1,
    STEPS + 1
):

    optimizer.zero_grad(
        set_to_none=True
    )

    outputs = model(
        image
    )

    loss, items = aqua_loss(
        outputs=outputs,
        targets=target,
        nc=NC,
        class_weights=None,
        box_weight=5.0,
        obj_weight=1.0,
        cls_weight=1.0,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0
    )

    optimizer.step()

    if (
        step == 1
        or step % 25 == 0
        or step == STEPS
    ):

        print(
            f"Step {step:3d}/{STEPS} "
            f"loss={loss.item():.4f} "
            f"box={items['box']:.4f} "
            f"obj={items['obj']:.4f} "
            f"cls={items['cls']:.4f}"
        )


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("OVERFIT INFERENCE")
print("=" * 70)

model.eval()

with torch.no_grad():

    outputs = model(
        image
    )

predictions = decode_predictions(
    outputs,
    conf_threshold=0.01,
    nms_iou=0.45
)[0]


print(
    "Predictions:",
    len(predictions["scores"])
)


# ============================================================
# TOP PREDICTIONS
# ============================================================

if len(predictions["scores"]) > 0:

    order = torch.argsort(
        predictions["scores"],
        descending=True
    )

    top_k = min(
        10,
        len(order)
    )

    for i in range(top_k):

        idx = order[i]

        score = float(
            predictions["scores"][idx]
            .detach()
            .cpu()
            .item()
        )

        label = int(
            predictions["labels"][idx]
            .detach()
            .cpu()
            .item()
        )

        box = (
            predictions["boxes"][idx]
            .detach()
            .cpu()
            .tolist()
        )

        print(
            f"{i + 1}. "
            f"class={label} "
            f"score={score:.4f} "
            f"box={[round(v, 3) for v in box]}"
        )

else:

    print(
        "No predictions."
    )


# ============================================================
# GROUND TRUTH BOX
# ============================================================

if target[0].numel() > 0:

    gt = target[0][0]

    gt_class = int(
        gt[0].item()
    )

    gt_box = xywh_to_xyxy(
        gt[1:5].reshape(1, 4)
    )[0]

    print()
    print(
        "First GT class:",
        gt_class
    )

    print(
        "First GT box:",
        [
            round(float(v), 3)
            for v in gt_box.detach().cpu()
        ]
    )


# ============================================================
# BEST MATCH
# ============================================================

if (
    len(predictions["scores"]) > 0
    and
    target[0].numel() > 0
):

    gt = target[0][0]

    gt_class = int(
        gt[0].item()
    )

    gt_box = xywh_to_xyxy(
        gt[1:5].reshape(1, 4)
    )[0]

    best_iou = 0.0
    best_score = 0.0
    best_class = -1

    for i in range(
        len(predictions["scores"])
    ):

        pred_class = int(
            predictions["labels"][i].item()
        )

        pred_box = predictions[
            "boxes"
        ][i]

        inter_x1 = max(
            float(pred_box[0]),
            float(gt_box[0])
        )

        inter_y1 = max(
            float(pred_box[1]),
            float(gt_box[1])
        )

        inter_x2 = min(
            float(pred_box[2]),
            float(gt_box[2])
        )

        inter_y2 = min(
            float(pred_box[3]),
            float(gt_box[3])
        )

        inter_w = max(
            0.0,
            inter_x2 - inter_x1
        )

        inter_h = max(
            0.0,
            inter_y2 - inter_y1
        )

        intersection = (
            inter_w
            *
            inter_h
        )

        pred_area = max(
            0.0,
            float(pred_box[2] - pred_box[0])
        ) * max(
            0.0,
            float(pred_box[3] - pred_box[1])
        )

        gt_area = max(
            0.0,
            float(gt_box[2] - gt_box[0])
        ) * max(
            0.0,
            float(gt_box[3] - gt_box[1])
        )

        union = (
            pred_area
            +
            gt_area
            -
            intersection
            +
            1e-9
        )

        iou = (
            intersection
            /
            union
        )

        if iou > best_iou:

            best_iou = iou

            best_score = float(
                predictions["scores"][i]
                .detach()
                .cpu()
                .item()
            )

            best_class = pred_class

    print()
    print("=" * 70)
    print("OVERFIT RESULT")
    print("=" * 70)

    print(
        "GT class:",
        gt_class
    )

    print(
        "Best predicted class:",
        best_class
    )

    print(
        "Best confidence:",
        round(best_score, 4)
    )

    print(
        "Best IoU:",
        round(best_iou, 4)
    )

    print("=" * 70)