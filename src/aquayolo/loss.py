import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BOX UTILITIES
# ============================================================

def xywh_to_xyxy(boxes):
    """
    Convert:
        [cx, cy, w, h]

    to:
        [x1, y1, x2, y2]

    Coordinates are normalized to the complete image.
    """

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    return torch.stack(
        [
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ],
        dim=1,
    )


def box_iou(box1, box2, eps=1e-7):
    """
    Pairwise IoU for XYXY boxes.
    """

    if box1.numel() == 0 or box2.numel() == 0:

        device = (
            box1.device
            if box1.numel() > 0
            else box2.device
        )

        return torch.zeros(
            (
                box1.shape[0],
                box2.shape[0],
            ),
            dtype=torch.float32,
            device=device,
        )

    area1 = (
        (box1[:, 2] - box1[:, 0]).clamp(min=0.0)
        *
        (box1[:, 3] - box1[:, 1]).clamp(min=0.0)
    )

    area2 = (
        (box2[:, 2] - box2[:, 0]).clamp(min=0.0)
        *
        (box2[:, 3] - box2[:, 1]).clamp(min=0.0)
    )

    lt = torch.maximum(
        box1[:, None, :2],
        box2[None, :, :2],
    )

    rb = torch.minimum(
        box1[:, None, 2:4],
        box2[None, :, 2:4],
    )

    wh = (
        rb - lt
    ).clamp(min=0.0)

    inter = (
        wh[..., 0]
        *
        wh[..., 1]
    )

    union = (
        area1[:, None]
        +
        area2[None, :]
        -
        inter
    )

    return inter / (
        union + eps
    )


# ============================================================
# SCALE ASSIGNMENT
# ============================================================

def choose_detection_scale(
    w,
    h,
):
    """
    Assign an object to one of the three detection scales.

    P3 -> small
    P4 -> medium
    P5 -> large
    """

    size_pixels = (
        (w * h) ** 0.5
    ) * 640.0

    if size_pixels < 64.0:
        return 0

    if size_pixels < 192.0:
        return 1

    return 2


# ============================================================
# TARGET BUILDING
# ============================================================

def build_targets(
    targets,
    batch_size,
    nc,
    grid_sizes,
):
    """
    Create one target assignment per object.

    Returns, for each scale:

        obj_target
        cls_target
        box_target
        box_mask
    """

    if len(targets) > 0:
        device = targets[0].device
    else:
        device = torch.device("cpu")

    results = []

    # --------------------------------------------------------
    # Empty scale tensors
    # --------------------------------------------------------

    for grid_h, grid_w in grid_sizes:

        obj_target = torch.zeros(
            batch_size,
            1,
            grid_h,
            grid_w,
            device=device,
        )

        cls_target = torch.full(
            (
                batch_size,
                grid_h,
                grid_w,
            ),
            -1,
            dtype=torch.long,
            device=device,
        )

        box_target = torch.zeros(
            batch_size,
            4,
            grid_h,
            grid_w,
            device=device,
        )

        box_mask = torch.zeros(
            batch_size,
            1,
            grid_h,
            grid_w,
            device=device,
        )

        results.append(
            (
                obj_target,
                cls_target,
                box_target,
                box_mask,
            )
        )

    # ========================================================
    # ASSIGN OBJECTS
    # ========================================================

    for batch_index in range(
        batch_size
    ):

        if batch_index >= len(targets):
            continue

        current_targets = targets[
            batch_index
        ]

        if (
            current_targets is None
            or current_targets.numel() == 0
        ):
            continue

        for row in current_targets:

            if row.numel() != 5:
                continue

            class_id = int(
                row[0].item()
            )

            if (
                class_id < 0
                or class_id >= nc
            ):
                continue

            cx = float(
                row[1].clamp(0.0, 1.0).item()
            )

            cy = float(
                row[2].clamp(0.0, 1.0).item()
            )

            w = float(
                row[3].clamp(0.0, 1.0).item()
            )

            h = float(
                row[4].clamp(0.0, 1.0).item()
            )

            if w <= 0.0 or h <= 0.0:
                continue

            # ------------------------------------------------
            # Select P3 / P4 / P5
            # ------------------------------------------------

            scale = choose_detection_scale(
                w,
                h,
            )

            grid_h, grid_w = grid_sizes[
                scale
            ]

            # ------------------------------------------------
            # Grid position
            # ------------------------------------------------

            gx = cx * grid_w
            gy = cy * grid_h

            gi = min(
                max(
                    int(gx),
                    0,
                ),
                grid_w - 1,
            )

            gj = min(
                max(
                    int(gy),
                    0,
                ),
                grid_h - 1,
            )

            (
                obj_target,
                cls_target,
                box_target,
                box_mask,
            ) = results[
                scale
            ]

            # ------------------------------------------------
            # Objectness target
            # ------------------------------------------------

            obj_target[
                batch_index,
                0,
                gj,
                gi,
            ] = 1.0

            # ------------------------------------------------
            # Class target
            # ------------------------------------------------

            cls_target[
                batch_index,
                gj,
                gi,
            ] = class_id

            # ------------------------------------------------
            # Box target
            # Full image normalized coordinates.
            # ------------------------------------------------

            box_target[
                batch_index,
                :,
                gj,
                gi,
            ] = torch.tensor(
                [
                    cx,
                    cy,
                    w,
                    h,
                ],
                dtype=torch.float32,
                device=device,
            )

            # ------------------------------------------------
            # Positive mask
            # ------------------------------------------------

            box_mask[
                batch_index,
                0,
                gj,
                gi,
            ] = 1.0

    return results


# ============================================================
# CLASS WEIGHTS
# ============================================================

def make_class_weights(
    class_counts,
    nc,
    min_weight=0.5,
    max_weight=4.0,
):
    """
    Inverse-frequency weights.

    Classes with zero training annotations receive weight 1.
    """

    weights = torch.ones(
        nc,
        dtype=torch.float32,
    )

    valid_counts = [
        float(v)
        for v in class_counts.values()
        if v > 0
    ]

    if not valid_counts:
        return weights

    valid_counts.sort()

    median_count = valid_counts[
        len(valid_counts) // 2
    ]

    for class_id in range(nc):

        count = class_counts.get(
            class_id,
            0,
        )

        if count > 0:

            weights[class_id] = (
                median_count
                /
                float(count)
            )

    return weights.clamp(
        min=min_weight,
        max=max_weight,
    )


# ============================================================
# HARD-NEGATIVE OBJECTNESS LOSS
# ============================================================

def balanced_objectness_loss(
    logits,
    targets,
):
    """
    Objectness loss with hard-negative mining.

    Positive cells:
        Fully supervised.

    Negative/background cells:
        Focus on the hardest background locations,
        i.e. cells where the model predicts the highest
        objectness.

    This is intended to reduce false detections on
    background regions such as stones, water haze,
    and other non-target areas.
    """

    positive_mask = (
        targets > 0.5
    )

    negative_mask = (
        targets <= 0.5
    )

    # --------------------------------------------------------
    # Positive object cells
    # --------------------------------------------------------

    if positive_mask.any():

        positive_loss = F.binary_cross_entropy_with_logits(
            logits[positive_mask],
            targets[positive_mask],
            reduction="mean",
        )

    else:

        positive_loss = torch.zeros(
            (),
            device=logits.device,
        )

    # --------------------------------------------------------
    # Negative/background cells
    # --------------------------------------------------------

    if negative_mask.any():

        negative_logits = logits[
            negative_mask
        ]

        negative_targets = targets[
            negative_mask
        ]

        negative_bce = F.binary_cross_entropy_with_logits(
            negative_logits,
            negative_targets,
            reduction="none",
        )

        # ----------------------------------------------------
        # Predicted objectness for background cells.
        # High probability means a hard negative.
        # ----------------------------------------------------

        negative_prob = torch.sigmoid(
            negative_logits
        )

        # ----------------------------------------------------
        # Select hardest 10% of background cells.
        #
        # At least 64 cells are considered when available.
        # ----------------------------------------------------

        total_negative = negative_prob.numel()
        num_pos = int(positive_mask.sum().item())

        # Standard OHEM 3:1 negative to positive ratio (bounded) to prevent
        # background penalties from overwhelming small underwater targets
        hard_k = max(16, min(max(3 * num_pos, 16), 128, total_negative))

        _, hard_indices = torch.topk(
            negative_prob,
            k=hard_k,
            largest=True,
            sorted=False,
        )

        hard_negative_loss = negative_bce[
            hard_indices
        ].mean()

    else:

        hard_negative_loss = torch.zeros(
            (),
            device=logits.device,
        )

    # --------------------------------------------------------
    # Combine positive and balanced hard-negative signals
    # --------------------------------------------------------

    return (
        positive_loss
        +
        0.5 * hard_negative_loss
    )


# ============================================================
# CLASSIFICATION LOSS — CLASS-WEIGHTED FOCAL LOSS
# ============================================================

def positive_classification_loss(
    logits,
    class_targets,
    class_weights=None,
    gamma=2.0,
):
    """
    Class-weighted focal loss computed only at positive
    object locations.

    Focal loss reduces the contribution of easy,
    already-correct classifications and focuses training
    on difficult examples.

    gamma:
        Focusing parameter.
        gamma=0 -> weighted cross entropy
        gamma=2 -> standard focal-loss setting

    Background cells do not contribute to classification.
    """

    positive_mask = (
        class_targets >= 0
    )

    if not positive_mask.any():

        return torch.zeros(
            (),
            device=logits.device,
            dtype=logits.dtype,
        )

    # --------------------------------------------------------
    # [B,C,H,W] -> [B,H,W,C]
    # --------------------------------------------------------

    logits = logits.permute(
        0,
        2,
        3,
        1,
    ).contiguous()

    positive_logits = logits[
        positive_mask
    ]

    positive_targets = class_targets[
        positive_mask
    ]

    # --------------------------------------------------------
    # Per-sample cross entropy.
    #
    # IMPORTANT:
    # reduction="none" lets us apply focal weighting
    # independently to every positive object.
    # --------------------------------------------------------

    ce_loss = F.cross_entropy(
        positive_logits,
        positive_targets,
        weight=class_weights,
        reduction="none",
    )

    # --------------------------------------------------------
    # Probability assigned to the correct class.
    # --------------------------------------------------------

    probabilities = F.softmax(
        positive_logits,
        dim=1,
    )

    target_probability = probabilities[
        torch.arange(
            positive_targets.shape[0],
            device=positive_targets.device,
        ),
        positive_targets,
    ]

    # --------------------------------------------------------
    # Focal modulation.
    #
    # Easy example:
    #   pt -> 1
    #   (1-pt)^gamma -> 0
    #
    # Hard example:
    #   pt -> 0
    #   (1-pt)^gamma -> 1
    # --------------------------------------------------------

    focal_factor = (
        1.0 - target_probability
    ).pow(gamma)

    focal_loss = (
        focal_factor
        *
        ce_loss
    )

    return focal_loss.mean()


# ============================================================
# AQUAYOLO LOSS
# ============================================================

def aqua_loss(
    outputs,
    targets,
    nc,
    class_weights=None,
    box_weight=5.0,
    obj_weight=1.0,
    cls_weight=1.0,
):
    """
    AquaYOLO training loss.

    Components:

        Box:
            Smooth-L1 + IoU

        Objectness:
            Positive BCE +
            hard-negative objectness mining

        Classification:
            Class-weighted focal loss only
            at positive object cells
    """

    if len(outputs) != 3:

        raise ValueError(
            "AquaYOLO requires exactly P3, P4 and P5."
        )

    batch_size = outputs[
        0
    ][
        "obj"
    ].shape[0]

    device = outputs[
        0
    ][
        "obj"
    ].device

    # --------------------------------------------------------
    # Grid sizes for P3 / P4 / P5
    # --------------------------------------------------------

    grid_sizes = [
        (
            outputs[i]["obj"].shape[-2],
            outputs[i]["obj"].shape[-1],
        )
        for i in range(3)
    ]

    target_scales = build_targets(
        targets=targets,
        batch_size=batch_size,
        nc=nc,
        grid_sizes=grid_sizes,
    )

    total_box = torch.zeros(
        (),
        device=device,
    )

    total_obj = torch.zeros(
        (),
        device=device,
    )

    total_cls = torch.zeros(
        (),
        device=device,
    )

    positive_scale_count = 0

    # ========================================================
    # P3 / P4 / P5
    # ========================================================

    for scale in range(3):

        pred = outputs[
            scale
        ]

        pred_obj = pred[
            "obj"
        ]

        pred_cls = pred[
            "cls"
        ]

        pred_box = pred[
            "box"
        ]

        (
            obj_target,
            cls_target,
            box_target,
            box_mask,
        ) = target_scales[
            scale
        ]

        # ====================================================
        # OBJECTNESS
        # ====================================================

        obj_loss = balanced_objectness_loss(
            pred_obj,
            obj_target,
        )

        total_obj = (
            total_obj
            +
            obj_loss
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if class_weights is not None:

            class_weights_for_device = (
                class_weights
                .to(
                    device=device,
                    dtype=pred_cls.dtype,
                )
            )

        else:

            class_weights_for_device = None

        cls_loss = positive_classification_loss(
            pred_cls,
            cls_target,
            class_weights_for_device,
        )

        total_cls = (
            total_cls
            +
            cls_loss
        )

        # ====================================================
        # BOXES
        # ====================================================

        positive_mask = (
            box_mask > 0.5
        )

        num_positive = int(
            positive_mask.sum().item()
        )

        if num_positive == 0:
            continue

        positive_scale_count += 1

        grid_h = pred_box.shape[
            -2
        ]

        grid_w = pred_box.shape[
            -1
        ]

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        gy, gx = torch.meshgrid(
            torch.arange(
                grid_h,
                device=device,
                dtype=pred_box.dtype,
            ),
            torch.arange(
                grid_w,
                device=device,
                dtype=pred_box.dtype,
            ),
            indexing="ij",
        )

        gx = gx.view(
            1,
            1,
            grid_h,
            grid_w,
        )

        gy = gy.view(
            1,
            1,
            grid_h,
            grid_w,
        )

        # ----------------------------------------------------
        # Decode predictions
        # ----------------------------------------------------

        pred_cx = (
            torch.sigmoid(
                pred_box[:, 0:1]
            )
            +
            gx
        ) / float(grid_w)

        pred_cy = (
            torch.sigmoid(
                pred_box[:, 1:2]
            )
            +
            gy
        ) / float(grid_h)

        pred_w = torch.sigmoid(
            pred_box[:, 2:3]
        )

        pred_h = torch.sigmoid(
            pred_box[:, 3:4]
        )

        pred_xywh = torch.cat(
            [
                pred_cx,
                pred_cy,
                pred_w,
                pred_h,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Select positive boxes
        # ----------------------------------------------------

        mask4 = box_mask.expand(
            -1,
            4,
            -1,
            -1,
        ).bool()

        predicted_boxes = pred_xywh[
            mask4
        ].reshape(
            -1,
            4,
        )

        target_boxes = box_target[
            mask4
        ].reshape(
            -1,
            4,
        )

        # ----------------------------------------------------
        # Smooth-L1
        # ----------------------------------------------------

        l1_loss = F.smooth_l1_loss(
            predicted_boxes,
            target_boxes,
            reduction="mean",
        )

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        predicted_xyxy = xywh_to_xyxy(
            predicted_boxes
        )

        target_xyxy = xywh_to_xyxy(
            target_boxes
        )

        iou_matrix = box_iou(
            predicted_xyxy,
            target_xyxy,
        )

        iou_values = torch.diagonal(
            iou_matrix,
        )

        iou_loss = (
            1.0
            -
            iou_values
        ).mean()

        box_loss = (
            0.5 * l1_loss
            +
            0.5 * iou_loss
        )

        total_box = (
            total_box
            +
            box_loss
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    total_obj = (
        total_obj
        /
        3.0
    )

    total_cls = (
        total_cls
        /
        3.0
    )

    if positive_scale_count > 0:

        total_box = (
            total_box
            /
            float(
                positive_scale_count
            )
        )

    else:

        total_box = (
            total_box
            *
            0.0
        )

    # ========================================================
    # TOTAL LOSS
    # ========================================================

    total_loss = (
        box_weight * total_box
        +
        obj_weight * total_obj
        +
        cls_weight * total_cls
    )

    items = {
        "box": float(
            total_box
            .detach()
            .cpu()
            .item()
        ),
        "obj": float(
            total_obj
            .detach()
            .cpu()
            .item()
        ),
        "cls": float(
            total_cls
            .detach()
            .cpu()
            .item()
        ),
    }

    return (
        total_loss,
        items,
    )