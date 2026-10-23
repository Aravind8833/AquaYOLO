import argparse
import csv
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from aquayolo.model import AquaYOLO
from aquayolo.train import (
    AquaYOLODataset,
    collate_fn,
    CLASS_NAMES,
    NUM_CLASSES,
)


# ============================================================
# BOX
# ============================================================

def xywh_to_xyxy(boxes):

    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    return torch.stack(
        [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ],
        dim=1,
    )


def box_iou(box1, box2):

    if box1.numel() == 0 or box2.numel() == 0:

        return torch.zeros(
            (
                box1.shape[0],
                box2.shape[0],
            ),
            device=(
                box1.device
                if box1.numel()
                else box2.device
            ),
        )

    area1 = (
        (box1[:, 2] - box1[:, 0]).clamp(min=0)
        *
        (box1[:, 3] - box1[:, 1]).clamp(min=0)
    )

    area2 = (
        (box2[:, 2] - box2[:, 0]).clamp(min=0)
        *
        (box2[:, 3] - box2[:, 1]).clamp(min=0)
    )

    lt = torch.maximum(
        box1[:, None, :2],
        box2[None, :, :2],
    )

    rb = torch.minimum(
        box1[:, None, 2:],
        box2[None, :, 2:],
    )

    wh = (
        rb - lt
    ).clamp(min=0)

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

    return inter / union.clamp(min=1e-9)


# ============================================================
# DECODER
# ============================================================

def decode_scale(
    prediction,
):
    """
    Decode one detection scale.

    Returns:
        boxes  [B,N,4]
        scores [B,N,C]
    """

    obj = prediction["obj"]
    cls = prediction["cls"]
    box = prediction["box"]

    bsz = obj.shape[0]
    h = obj.shape[2]
    w = obj.shape[3]

    nc = cls.shape[1]
    device = obj.device

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    gy, gx = torch.meshgrid(
        torch.arange(
            h,
            device=device,
            dtype=torch.float32,
        ),
        torch.arange(
            w,
            device=device,
            dtype=torch.float32,
        ),
        indexing="ij",
    )

    gx = gx.view(1, h, w)
    gy = gy.view(1, h, w)

    # --------------------------------------------------------
    # Box decoding
    # --------------------------------------------------------

    cx = (
        torch.sigmoid(box[:, 0])
        + gx
    ) / w

    cy = (
        torch.sigmoid(box[:, 1])
        + gy
    ) / h

    bw = torch.sigmoid(
        box[:, 2]
    )

    bh = torch.sigmoid(
        box[:, 3]
    )

    boxes = torch.stack(
        [
            cx,
            cy,
            bw,
            bh,
        ],
        dim=-1,
    )

    boxes = boxes.reshape(
        bsz,
        -1,
        4,
    )

    # --------------------------------------------------------
    # Objectness
    # --------------------------------------------------------

    obj_prob = torch.sigmoid(
        obj
    ).permute(
        0,
        2,
        3,
        1,
    ).contiguous().reshape(
        bsz,
        -1,
        1,
    )

    # --------------------------------------------------------
    # CLASS PROBABILITY
    #
    # IMPORTANT:
    # CrossEntropy-trained class logits require SOFTMAX.
    # Do NOT use sigmoid here.
    # --------------------------------------------------------

    cls_prob = torch.softmax(
        cls,
        dim=1,
    ).permute(
        0,
        2,
        3,
        1,
    ).contiguous().reshape(
        bsz,
        -1,
        nc,
    )

    # --------------------------------------------------------
    # Final confidence
    # --------------------------------------------------------

    scores = (
        obj_prob
        *
        cls_prob
    )

    return boxes, scores


def decode_predictions(
    outputs,
    conf_threshold=0.10,
    nms_iou=0.45,
):

    box_list = []
    score_list = []

    for output in outputs:

        boxes, scores = decode_scale(
            output
        )

        box_list.append(
            boxes
        )

        score_list.append(
            scores
        )

    boxes = torch.cat(
        box_list,
        dim=1,
    )

    scores = torch.cat(
        score_list,
        dim=1,
    )

    results = []

    for b in range(
        boxes.shape[0]
    ):

        b_boxes = boxes[b]
        b_scores = scores[b]

        confidence, labels = torch.max(
            b_scores,
            dim=1,
        )

        keep = (
            confidence >= conf_threshold
        )

        b_boxes = b_boxes[keep]
        confidence = confidence[keep]
        labels = labels[keep]

        if b_boxes.numel() == 0:

            results.append(
                {
                    "boxes": torch.empty(
                        (0, 4),
                        device=boxes.device,
                    ),
                    "scores": torch.empty(
                        (0,),
                        device=boxes.device,
                    ),
                    "labels": torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=boxes.device,
                    ),
                }
            )

            continue

        b_boxes = xywh_to_xyxy(
            b_boxes
        )

        b_boxes[:, 0::2] = b_boxes[
            :, 0::2
        ].clamp(
            0,
            1,
        )

        b_boxes[:, 1::2] = b_boxes[
            :, 1::2
        ].clamp(
            0,
            1,
        )

        valid = (
            (b_boxes[:, 2] > b_boxes[:, 0])
            &
            (b_boxes[:, 3] > b_boxes[:, 1])
        )

        b_boxes = b_boxes[valid]
        confidence = confidence[valid]
        labels = labels[valid]

        if b_boxes.numel() == 0:

            results.append(
                {
                    "boxes": torch.empty(
                        (0, 4),
                        device=boxes.device,
                    ),
                    "scores": torch.empty(
                        (0,),
                        device=boxes.device,
                    ),
                    "labels": torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=boxes.device,
                    ),
                }
            )

            continue


# Class-aware NMS.
# Suppress overlapping boxes only when they have
# the same predicted class.
        keep_indices = torchvision.ops.batched_nms(
            b_boxes,
            confidence,
            labels,
            nms_iou,
        )


        order = torch.argsort(
            confidence[
                keep_indices
            ],
            descending=True,
        )

        keep_indices = keep_indices[
            order
        ]
        MAX_DETECTIONS = 50

        if keep_indices.numel() > MAX_DETECTIONS:
            keep_indices = keep_indices[:MAX_DETECTIONS]

        results.append(
            {
                "boxes": b_boxes[
                    keep_indices
                ],
                "scores": confidence[
                    keep_indices
                ],
                "labels": labels[
                    keep_indices
                ],
            }
        )

    return results


# ============================================================
# MATCHING
# ============================================================


def collect_stats(
    predictions,
    targets,
    iou_threshold,
):
    stats = {
        i: {
            "scores": [],
            "matches": [],
            "num_gt": 0,
        }
        for i in range(NUM_CLASSES)
    }

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        if target is not None and target.numel() > 0:
            gt_labels = target[:, 0].long()
            gt_boxes = xywh_to_xyxy(target[:, 1:5])
        else:
            gt_labels = torch.empty(
                (0,), dtype=torch.long
            )
            gt_boxes = torch.empty((0, 4), dtype=torch.float32)

        for cls_id in range(NUM_CLASSES):
            gt_mask = gt_labels == cls_id
            pred_mask = pred_labels == cls_id

            gtb = gt_boxes[gt_mask]
            pb = pred_boxes[pred_mask]
            ps = pred_scores[pred_mask]

            stats[cls_id]["num_gt"] += gtb.shape[0]

            if pb.shape[0] == 0:
                continue

            if gtb.shape[0] == 0:
                stats[cls_id]["scores"].extend(
                    ps.detach().cpu().tolist()
                )
                stats[cls_id]["matches"].extend(
                    [False] * pb.shape[0]
                )
                continue

            ious = box_iou(pb, gtb)
            order = torch.argsort(ps, descending=True)
            matched_gt = set()

            # Process predictions from highest to lowest confidence.
            for pidx in order.tolist():
                # Ignore ground-truth boxes already assigned to
                # a higher-confidence prediction.
                candidate_ious = ious[pidx].clone()

                if matched_gt:
                    candidate_ious[list(matched_gt)] = -1.0

                best_iou, best_gt = torch.max(
                    candidate_ious, dim=0
                )

                best_iou = float(best_iou.detach().cpu().item())
                best_gt = int(best_gt.detach().cpu().item())
                matched = best_iou >= iou_threshold

                stats[cls_id]["scores"].append(
                    float(ps[pidx].detach().cpu().item())
                )
                stats[cls_id]["matches"].append(bool(matched))

                if matched:
                    matched_gt.add(best_gt)

    return stats


# ============================================================
# AP
# ============================================================

def compute_ap(
    scores,
    matches,
    num_gt,
):

    if num_gt <= 0:
        return None

    if len(scores) == 0:
        return 0.0

    scores = torch.tensor(
        scores,
        dtype=torch.float64,
    )

    matches = torch.tensor(
        matches,
        dtype=torch.float64,
    )

    order = torch.argsort(
        scores,
        descending=True,
    )

    matches = matches[
        order
    ]

    tp = torch.cumsum(
        matches,
        dim=0,
    )

    fp = torch.cumsum(
        1.0 - matches,
        dim=0,
    )

    recall = tp / float(
        max(num_gt, 1)
    )

    precision = tp / (
        tp + fp
    ).clamp(
        min=1e-12
    )

    ap = 0.0

    for r in torch.linspace(
        0,
        1,
        101,
        dtype=torch.float64,
    ):

        valid = precision[
            recall >= r
        ]

        if valid.numel():
            ap += float(
                valid.max().item()
            )

    return ap / 101.0


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    conf,
    nms_iou,
):

    model.eval()

    all_predictions = []
    all_targets = []

    for step, (
        images,
        targets,
    ) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        outputs = model(
            images
        )

        predictions = decode_predictions(
            outputs,
            conf_threshold=conf,
            nms_iou=nms_iou,
        )

        for p in predictions:

            all_predictions.append(
                {
                    "boxes":
                        p["boxes"].cpu(),
                    "scores":
                        p["scores"].cpu(),
                    "labels":
                        p["labels"].cpu(),
                }
            )

        all_targets.extend(
            [
                t.cpu()
                for t in targets
            ]
        )

        if (
            step == 1
            or step % 25 == 0
            or step == len(loader)
        ):

            print(
                f"  Evaluated "
                f"{step}/{len(loader)} batches"
            )

    iou_thresholds = [
        0.50 + 0.05 * i
        for i in range(10)
    ]

    results = {
        i: {
            "name": CLASS_NAMES[i],
            "num_gt": 0,
            "precision": 0.0,
            "recall": 0.0,
            "ap50": 0.0,
            "ap5095": 0.0,
        }
        for i in range(NUM_CLASSES)
    }

    for threshold in iou_thresholds:

        stats = collect_stats(
            all_predictions,
            all_targets,
            threshold,
        )

        for cls_id in range(NUM_CLASSES):

            s = stats[cls_id]

            results[cls_id][
                "num_gt"
            ] = s["num_gt"]

            ap = compute_ap(
                s["scores"],
                s["matches"],
                s["num_gt"],
            )

            if ap is None:
                continue

            results[cls_id][
                "ap5095"
            ] += (
                ap / len(iou_thresholds)
            )

            if abs(
                threshold - 0.50
            ) < 1e-9:

                results[cls_id][
                    "ap50"
                ] = ap

                tp = sum(
                    bool(x)
                    for x in s["matches"]
                )

                fp = (
                    len(s["matches"])
                    -
                    tp
                )

                results[cls_id][
                    "precision"
                ] = (
                    tp
                    /
                    max(tp + fp, 1)
                )

                results[cls_id][
                    "recall"
                ] = (
                    tp
                    /
                    max(s["num_gt"], 1)
                )

    active = [
        i
        for i in range(NUM_CLASSES)
        if results[i]["num_gt"] > 0
    ]

    if active:

        map50 = sum(
            results[i]["ap50"]
            for i in active
        ) / len(active)

        map5095 = sum(
            results[i]["ap5095"]
            for i in active
        ) / len(active)

        precision = sum(
            results[i]["precision"]
            for i in active
        ) / len(active)

        recall = sum(
            results[i]["recall"]
            for i in active
        ) / len(active)

    else:

        map50 = 0.0
        map5095 = 0.0
        precision = 0.0
        recall = 0.0

    return {
        "images":
            len(all_targets),
        "mAP50":
            map50,
        "mAP5095":
            map5095,
        "precision50":
            precision,
        "recall50":
            recall,
        "per_class":
            results,
        "active":
            active,
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_results(
    output_dir,
    results,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = (
        output_dir
        /
        "evaluation_summary.csv"
    )

    per_class = (
        output_dir
        /
        "per_class_results.csv"
    )

    with open(
        summary,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            ["metric", "value"]
        )

        w.writerow(
            ["images", results["images"]]
        )

        w.writerow(
            ["mAP50", results["mAP50"]]
        )

        w.writerow(
            ["mAP50_95", results["mAP5095"]]
        )

        w.writerow(
            ["precision50", results["precision50"]]
        )

        w.writerow(
            ["recall50", results["recall50"]]
        )

    with open(
        per_class,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
                "class_id",
                "class_name",
                "ground_truth",
                "precision50",
                "recall50",
                "AP50",
                "AP50_95",
            ]
        )

        for cls_id in range(NUM_CLASSES):

            r = results[
                "per_class"
            ][cls_id]

            w.writerow(
                [
                    cls_id,
                    r["name"],
                    r["num_gt"],
                    r["precision"],
                    r["recall"],
                    r["ap50"],
                    r["ap5095"],
                ]
            )

    return summary, per_class


# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--weights",
        required=True,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for evaluation (0.001 matches standard VOC/COCO AP calculation)",
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--out",
        default="./runs/evaluation",
    )
    
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print(f"AquaYOLO {args.split.upper()} EVALUATION")
    print("=" * 70)

    print(
        "Device       :",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU          :",
            torch.cuda.get_device_name(0),
        )

    print(
        "Checkpoint   :",
        args.weights,
    )

    dataset = AquaYOLODataset(
        Path(args.data),
        args.split,
        args.size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = AquaYOLO(
        nc=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        args.weights,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    print(
        "Checkpoint loaded successfully."
    )

    results = evaluate(
        model,
        loader,
        device,
        args.conf,
        args.nms_iou,
    )

    summary, per_class = save_results(
        Path(args.out),
        results,
    )

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Images          : {results['images']}"
    )

    print(
        f"mAP@0.50        : {results['mAP50']:.6f}"
    )

    print(
        f"mAP@0.50:0.95   : {results['mAP5095']:.6f}"
    )

    print(
        f"Precision@0.50  : {results['precision50']:.6f}"
    )

    print(
        f"Recall@0.50     : {results['recall50']:.6f}"
    )

    print()
    print(
        "Summary:",
        summary,
    )

    print(
        "Per class:",
        per_class,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()