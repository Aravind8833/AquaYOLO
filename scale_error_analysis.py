from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from aquayolo.model import AquaYOLO
from aquayolo.train import AquaYOLODataset
from aquayolo.evaluate import decode_scale
from aquayolo.loss import (
    xywh_to_xyxy,
    choose_detection_scale,
)


NC = 26

SCALE_NAMES = {
    0: "P3",
    1: "P4",
    2: "P5",
}


def collate_fn(batch):
    images = torch.stack(
        [item[0] for item in batch],
        dim=0,
    )

    targets = [
        item[1]
        for item in batch
    ]

    return images, targets


def load_checkpoint(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )


def decode_with_scale(outputs, conf_threshold):
    """
    Decode P3/P4/P5 separately and preserve the
    source detection scale for every candidate.
    """

    all_boxes = []
    all_scores = []
    all_labels = []
    all_scales = []

    raw_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    for scale_index, output in enumerate(outputs):

        boxes, scores = decode_scale(output)

        # One class per candidate.
        confidence, labels = torch.max(
            scores,
            dim=2,
        )

        confidence = confidence[0]
        labels = labels[0]
        boxes = boxes[0]

        keep = confidence >= conf_threshold

        boxes = boxes[keep]
        confidence = confidence[keep]
        labels = labels[keep]

        raw_counts[scale_index] = int(
            keep.sum().item()
        )

        if boxes.numel() == 0:
            continue

        boxes = xywh_to_xyxy(boxes)

        boxes[:, 0::2] = boxes[
            :, 0::2
        ].clamp(0, 1)

        boxes[:, 1::2] = boxes[
            :, 1::2
        ].clamp(0, 1)

        valid = (
            (boxes[:, 2] > boxes[:, 0])
            &
            (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[valid]
        confidence = confidence[valid]
        labels = labels[valid]

        if boxes.numel() == 0:
            continue

        scale_tensor = torch.full(
            (boxes.shape[0],),
            scale_index,
            dtype=torch.long,
            device=boxes.device,
        )

        all_boxes.append(boxes)
        all_scores.append(confidence)
        all_labels.append(labels)
        all_scales.append(scale_tensor)

    if not all_boxes:
        return (
            torch.empty(
                (0, 4),
                device=outputs[0]["obj"].device,
            ),
            torch.empty(
                (0,),
                device=outputs[0]["obj"].device,
            ),
            torch.empty(
                (0,),
                dtype=torch.long,
                device=outputs[0]["obj"].device,
            ),
            torch.empty(
                (0,),
                dtype=torch.long,
                device=outputs[0]["obj"].device,
            ),
            raw_counts,
        )

    return (
        torch.cat(all_boxes, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_scales, dim=0),
        raw_counts,
    )


def apply_nms(
    boxes,
    scores,
    labels,
    scales,
    nms_iou=0.45,
    max_detections=20,
):
    if boxes.numel() == 0:
        return (
            boxes,
            scores,
            labels,
            scales,
        )

    # Keep exactly the same class-agnostic NMS
    # behavior used by the current evaluator.
    keep = torchvision.ops.nms(
        boxes,
        scores,
        nms_iou,
    )

    order = torch.argsort(
        scores[keep],
        descending=True,
    )

    keep = keep[order]

    if keep.numel() > max_detections:
        keep = keep[:max_detections]

    return (
        boxes[keep],
        scores[keep],
        labels[keep],
        scales[keep],
    )


def iou_one_to_many(box, boxes):
    if boxes.numel() == 0:
        return torch.empty(
            (0,),
            dtype=box.dtype,
            device=box.device,
        )

    x1 = torch.maximum(
        box[0],
        boxes[:, 0],
    )

    y1 = torch.maximum(
        box[1],
        boxes[:, 1],
    )

    x2 = torch.minimum(
        box[2],
        boxes[:, 2],
    )

    y2 = torch.minimum(
        box[3],
        boxes[:, 3],
    )

    inter = (
        (x2 - x1).clamp(min=0)
        *
        (y2 - y1).clamp(min=0)
    )

    area_a = (
        (box[2] - box[0]).clamp(min=0)
        *
        (box[3] - box[1]).clamp(min=0)
    )

    area_b = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        *
        (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )

    union = area_a + area_b - inter

    return inter / union.clamp(min=1e-9)


def match_predictions(
    pred_boxes,
    pred_scores,
    pred_labels,
    pred_scales,
    target,
    iou_threshold=0.50,
):
    target = target.to(pred_boxes.device)
        
    """
    Match detections to GT in descending confidence order.
    """

    gt_boxes = xywh_to_xyxy(
        target[:, 1:5].float()
    )

    gt_labels = target[:, 0].long()

    if gt_boxes.numel() == 0:
        gt_boxes = torch.empty(
            (0, 4)
        )
        gt_labels = torch.empty(
            (0,),
            dtype=torch.long,
        )

    matched_gt = set()

    tp_by_scale = {
        0: 0,
        1: 0,
        2: 0,
    }

    fp_by_scale = {
        0: 0,
        1: 0,
        2: 0,
    }

    confusion_by_scale = {
        0: 0,
        1: 0,
        2: 0,
    }

    confusion_gt_scale = {
        0: 0,
        1: 0,
        2: 0,
    }

    if pred_scores.numel() > 0:
        order = torch.argsort(
            pred_scores,
            descending=True,
        )
    else:
        order = torch.empty(
            (0,),
            dtype=torch.long,
        )

    for pred_index in order.tolist():

        p_box = pred_boxes[pred_index]
        p_label = int(
            pred_labels[pred_index]
        )
        p_scale = int(
            pred_scales[pred_index]
        )

        best_same = None
        best_same_iou = 0.0

        best_other = None
        best_other_iou = 0.0

        for gt_index in range(
            len(gt_boxes)
        ):

            if gt_index in matched_gt:
                continue

            iou = float(
                iou_one_to_many(
                    p_box,
                    gt_boxes[
                        gt_index
                    ].unsqueeze(0),
                )[0]
            )

            gt_label = int(
                gt_labels[gt_index]
            )

            if (
                gt_label == p_label
                and iou >= iou_threshold
                and iou > best_same_iou
            ):
                best_same = gt_index
                best_same_iou = iou

            elif (
                gt_label != p_label
                and iou >= iou_threshold
                and iou > best_other_iou
            ):
                best_other = gt_index
                best_other_iou = iou

        if best_same is not None:

            matched_gt.add(
                best_same
            )

            tp_by_scale[p_scale] += 1

        elif best_other is not None:

            matched_gt.add(
                best_other
            )

            confusion_by_scale[p_scale] += 1

            gt_w = float(
                target[
                    best_other,
                    3,
                ].item()
            )

            gt_h = float(
                target[
                    best_other,
                    4,
                ].item()
            )

            gt_scale = choose_detection_scale(
                gt_w,
                gt_h,
            )

            confusion_gt_scale[
                gt_scale
            ] += 1

        else:

            fp_by_scale[p_scale] += 1

    # Unmatched GT = false negatives.
    fn_by_gt_scale = {
        0: 0,
        1: 0,
        2: 0,
    }

    for gt_index in range(
        len(gt_boxes)
    ):

        if gt_index in matched_gt:
            continue

        w = float(
            target[
                gt_index,
                3,
            ].item()
        )

        h = float(
            target[
                gt_index,
                4,
            ].item()
        )

        expected_scale = choose_detection_scale(
            w,
            h,
        )

        fn_by_gt_scale[
            expected_scale
        ] += 1

    return {
        "tp": tp_by_scale,
        "fp": fp_by_scale,
        "confusion": confusion_by_scale,
        "confusion_gt": confusion_gt_scale,
        "fn": fn_by_gt_scale,
    }


def main():

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
        default=4,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--out",
        default=(
            ".\\runs\\"
            "aquayolo_fishinv_pretrained_30epoch\\"
            "error_analysis\\"
            "scale_error_analysis.csv"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_root = Path(
        args.data
    )

    print("=" * 70)
    print("AquaYOLO SCALE ERROR ANALYSIS")
    print("=" * 70)

    print(
        "Dataset :",
        data_root.resolve(),
    )

    print(
        "Weights :",
        Path(args.weights).resolve(),
    )

    print(
        "Device  :",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU     :",
            torch.cuda.get_device_name(0),
        )

    dataset = AquaYOLODataset(
        root=data_root,
        split="test",
        size=args.size,
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
        nc=NC
    ).to(device)

    load_checkpoint(
        model,
        Path(args.weights),
        device,
    )

    model.eval()

    total = {
        scale: {
            "raw": 0,
            "kept": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "confusion": 0,
            "confusion_gt": 0,
        }
        for scale in range(3)
    }

    total_images = 0

    with torch.no_grad():

        for batch_index, (
            images,
            targets,
        ) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            outputs = model(
                images
            )

            for local_index in range(
                len(targets)
            ):

                single_outputs = [
                    {
                        key: value[
                            local_index:
                            local_index + 1
                        ]
                        for key, value in output.items()
                    }
                    for output in outputs
                ]

                (
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    pred_scales,
                    raw_counts,
                ) = decode_with_scale(
                    single_outputs,
                    args.conf,
                )

                for scale in range(3):

                    total[
                        scale
                    ]["raw"] += raw_counts[
                        scale
                    ]

                (
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    pred_scales,
                ) = apply_nms(
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    pred_scales,
                    nms_iou=args.nms_iou,
                    max_detections=20,
                )

                for scale in range(3):

                    total[
                        scale
                    ]["kept"] += int(
                        (
                            pred_scales
                            == scale
                        ).sum().item()
                    )

                result = match_predictions(
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    pred_scales,
                    targets[local_index],
                    iou_threshold=args.match_iou,
                )

                for scale in range(3):

                    total[
                        scale
                    ]["tp"] += result[
                        "tp"
                    ][scale]

                    total[
                        scale
                    ]["fp"] += result[
                        "fp"
                    ][scale]

                    total[
                        scale
                    ]["fn"] += result[
                        "fn"
                    ][scale]

                    total[
                        scale
                    ]["confusion"] += result[
                        "confusion"
                    ][scale]

                    total[
                        scale
                    ]["confusion_gt"] += result[
                        "confusion_gt"
                    ][scale]

                total_images += 1

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or (batch_index + 1) == len(loader)
            ):

                print(
                    f"Processed "
                    f"{min(total_images, len(dataset))}"
                    f"/{len(dataset)} images"
                )

    rows = []

    print()
    print("=" * 70)
    print("SCALE RESULTS")
    print("=" * 70)

    for scale in range(3):

        item = total[
            scale
        ]

        tp = item["tp"]
        fp = item["fp"]
        fn = item["fn"]
        confusion = item["confusion"]

        precision = (
            tp
            / (
                tp
                + fp
                + confusion
            )
            if (
                tp
                + fp
                + confusion
            ) > 0
            else 0.0
        )

        recall = (
            tp
            / (
                tp
                + fn
                + confusion
            )
            if (
                tp
                + fn
                + confusion
            ) > 0
            else 0.0
        )

        raw_per_image = (
            item["raw"]
            / total_images
        )

        kept_per_image = (
            item["kept"]
            / total_images
        )

        print()
        print(
            SCALE_NAMES[scale]
        )

        print(
            "  Raw candidates    :",
            item["raw"],
            f"({raw_per_image:.2f}/image)",
        )

        print(
            "  After NMS         :",
            item["kept"],
            f"({kept_per_image:.2f}/image)",
        )

        print(
            "  TP                :",
            tp,
        )

        print(
            "  FP                :",
            fp,
        )

        print(
            "  FN                :",
            fn,
        )

        print(
            "  Class confusions  :",
            confusion,
        )

        print(
            "  Precision-like    :",
            f"{precision:.4f}",
        )

        print(
            "  Recall-like       :",
            f"{recall:.4f}",
        )

        rows.append(
            {
                "scale": SCALE_NAMES[scale],
                "raw_candidates": item["raw"],
                "kept_after_nms": item["kept"],
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "class_confusions": confusion,
                "confusion_gt_scale": item[
                    "confusion_gt"
                ],
                "raw_per_image": raw_per_image,
                "kept_per_image": kept_per_image,
                "precision_like": precision,
                "recall_like": recall,
            }
        )

    out_path = Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "Saved:",
        out_path.resolve(),
    )


if __name__ == "__main__":
    main()