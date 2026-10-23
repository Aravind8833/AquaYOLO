from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from aquayolo.model import AquaYOLO
from aquayolo.train import AquaYOLODataset
from aquayolo.evaluate import decode_scale
from aquayolo.loss import xywh_to_xyxy


CLASS_NAMES = [
    "jellyfish",
    "penguin",
    "puffin",
    "starfish",
    "stingray",
    "basking_shark",
    "blacktip_reef_shark",
    "great_white_shark",
    "hammerhead_shark",
    "nurse_shark",
    "oceanic_whitetip_shark",
    "thresher_shark",
    "whale_shark",
    "whitetip_reef_shark",
    "ballan_wrasse",
    "cuckoo_wrasse",
    "corkwing_wrasse",
    "goldsinny_wrasse",
    "rock_cook",
    "atlantic_cod",
    "pollack",
    "whiting",
    "spiny_dogfish",
    "brown_crab",
    "two_spot_goby",
    "human",
]

NC = len(CLASS_NAMES)


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def load_checkpoint(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)


def decode_raw_outputs(outputs):
    boxes_all = []
    scores_all = []

    for output in outputs:
        boxes, scores = decode_scale(output)
        boxes_all.append(boxes)
        scores_all.append(scores)

    boxes = torch.cat(boxes_all, dim=1)
    scores = torch.cat(scores_all, dim=1)

    return boxes, scores


def make_predictions(
    boxes,
    scores,
    conf_threshold,
    nms_iou,
    class_aware,
    max_detections=20,
):
    results = []

    batch_size = boxes.shape[0]

    for b in range(batch_size):
        b_boxes = boxes[b]
        b_scores = scores[b]

        confidence, labels = torch.max(
            b_scores,
            dim=1,
        )

        keep = confidence >= conf_threshold

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

        b_boxes = xywh_to_xyxy(b_boxes)

        b_boxes[:, 0::2] = b_boxes[:, 0::2].clamp(0, 1)
        b_boxes[:, 1::2] = b_boxes[:, 1::2].clamp(0, 1)

        valid = (
            (b_boxes[:, 2] > b_boxes[:, 0])
            & (b_boxes[:, 3] > b_boxes[:, 1])
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

        if class_aware:
            keep_indices = torchvision.ops.batched_nms(
                b_boxes,
                confidence,
                labels,
                nms_iou,
            )
        else:
            keep_indices = torchvision.ops.nms(
                b_boxes,
                confidence,
                nms_iou,
            )

        order = torch.argsort(
            confidence[keep_indices],
            descending=True,
        )

        keep_indices = keep_indices[order]

        if keep_indices.numel() > max_detections:
            keep_indices = keep_indices[:max_detections]

        results.append(
            {
                "boxes": b_boxes[keep_indices],
                "scores": confidence[keep_indices],
                "labels": labels[keep_indices],
            }
        )

    return results


def box_iou_one_to_many(box, boxes):
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

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    box_area = (
        (box[2] - box[0]).clamp(min=0)
        * (box[3] - box[1]).clamp(min=0)
    )

    boxes_area = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )

    union = box_area + boxes_area - inter

    return inter / union.clamp(min=1e-9)


def target_to_xyxy(target):
    if target.numel() == 0:
        return (
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )

    labels = target[:, 0].long()
    boxes = xywh_to_xyxy(target[:, 1:5].float())

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 1)

    return boxes, labels


def analyze_image(prediction, target, iou_threshold=0.50):
    pred_boxes = prediction["boxes"].detach().cpu()
    pred_scores = prediction["scores"].detach().cpu()
    pred_labels = prediction["labels"].detach().cpu()

    gt_boxes, gt_labels = target_to_xyxy(
        target.detach().cpu()
    )

    unmatched_gt = set(range(len(gt_boxes)))

    true_positives = []
    false_positives = []
    confusions = []
    gt_confused = set()

    if len(pred_scores) > 0:
        order = torch.argsort(
            pred_scores,
            descending=True,
        ).tolist()
    else:
        order = []

    for pred_index in order:
        p_box = pred_boxes[pred_index]
        p_label = int(pred_labels[pred_index])
        p_score = float(pred_scores[pred_index])

        candidates = sorted(
            list(unmatched_gt),
            key=lambda gi: float(
                box_iou_one_to_many(
                    p_box,
                    gt_boxes[gi].unsqueeze(0),
                )[0]
            ),
            reverse=True,
        )

        same_class_match = None
        same_class_iou = 0.0

        for gi in candidates:
            if int(gt_labels[gi]) != p_label:
                continue

            iou = float(
                box_iou_one_to_many(
                    p_box,
                    gt_boxes[gi].unsqueeze(0),
                )[0]
            )

            if iou >= iou_threshold:
                same_class_match = gi
                same_class_iou = iou
                break

        if same_class_match is not None:
            unmatched_gt.remove(same_class_match)

            true_positives.append(
                {
                    "pred_index": pred_index,
                    "gt_index": same_class_match,
                    "score": p_score,
                    "iou": same_class_iou,
                    "class_id": p_label,
                }
            )
            continue

        # Look for a high-IoU ground truth of a different class.
        confusion_match = None
        confusion_iou = 0.0

        for gi in candidates:
            gt_label = int(gt_labels[gi])

            if gt_label == p_label:
                continue

            iou = float(
                box_iou_one_to_many(
                    p_box,
                    gt_boxes[gi].unsqueeze(0),
                )[0]
            )

            if iou >= iou_threshold:
                confusion_match = gi
                confusion_iou = iou
                break

        if confusion_match is not None:
            unmatched_gt.remove(confusion_match)
            gt_confused.add(confusion_match)

            confusions.append(
                {
                    "pred_index": pred_index,
                    "gt_index": confusion_match,
                    "score": p_score,
                    "iou": confusion_iou,
                    "pred_class": p_label,
                    "gt_class": int(gt_labels[confusion_match]),
                }
            )
            continue

        false_positives.append(
            {
                "pred_index": pred_index,
                "score": p_score,
                "pred_class": p_label,
            }
        )

    false_negatives = []

    for gi in sorted(unmatched_gt):
        false_negatives.append(
            {
                "gt_index": gi,
                "gt_class": int(gt_labels[gi]),
            }
        )

    return {
        "tp": true_positives,
        "fp": false_positives,
        "confusions": confusions,
        "fn": false_negatives,
        "pred_boxes": pred_boxes,
        "pred_scores": pred_scores,
        "pred_labels": pred_labels,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
    }


def draw_text(draw, xy, text, fill):
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text(
        xy,
        text,
        fill=fill,
        font=font,
    )


def draw_analysis_image(
    image_path,
    analysis,
    output_path,
):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((640, 640))

    draw = ImageDraw.Draw(image)

    gt_boxes = analysis["gt_boxes"]
    gt_labels = analysis["gt_labels"]

    for gi in range(len(gt_boxes)):
        box = gt_boxes[gi].tolist()
        label = int(gt_labels[gi])

        x1 = int(box[0] * 640)
        y1 = int(box[1] * 640)
        x2 = int(box[2] * 640)
        y2 = int(box[3] * 640)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 255, 0),
            width=2,
        )

        draw_text(
            draw,
            (x1, max(0, y1 - 12)),
            f"GT: {CLASS_NAMES[label]}",
            (0, 255, 0),
        )

    for item in analysis["tp"]:
        pi = item["pred_index"]
        box = analysis["pred_boxes"][pi].tolist()
        label = item["class_id"]
        score = item["score"]

        x1 = int(box[0] * 640)
        y1 = int(box[1] * 640)
        x2 = int(box[2] * 640)
        y2 = int(box[3] * 640)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 128, 255),
            width=2,
        )

        draw_text(
            draw,
            (x1, min(628, y2 + 2)),
            f"TP: {CLASS_NAMES[label]} {score:.2f}",
            (0, 128, 255),
        )

    for item in analysis["fp"]:
        pi = item["pred_index"]
        box = analysis["pred_boxes"][pi].tolist()
        label = item["pred_class"]
        score = item["score"]

        x1 = int(box[0] * 640)
        y1 = int(box[1] * 640)
        x2 = int(box[2] * 640)
        y2 = int(box[3] * 640)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 0, 0),
            width=3,
        )

        draw_text(
            draw,
            (x1, y1),
            f"FP: {CLASS_NAMES[label]} {score:.2f}",
            (255, 0, 0),
        )

    for item in analysis["confusions"]:
        pi = item["pred_index"]
        box = analysis["pred_boxes"][pi].tolist()

        pred_label = item["pred_class"]
        gt_label = item["gt_class"]
        score = item["score"]

        x1 = int(box[0] * 640)
        y1 = int(box[1] * 640)
        x2 = int(box[2] * 640)
        y2 = int(box[3] * 640)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 165, 0),
            width=3,
        )

        draw_text(
            draw,
            (x1, y1),
            (
                f"CONF: {CLASS_NAMES[gt_label]}"
                f" -> {CLASS_NAMES[pred_label]}"
                f" {score:.2f}"
            ),
            (255, 165, 0),
        )

    image.save(output_path)


def write_csv(path, fieldnames, rows):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="AquaYOLO test-set error analysis"
    )

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
            "error_analysis"
        ),
    )

    args = parser.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    agnostic_dir = out_root / "class_agnostic"
    aware_dir = out_root / "class_aware"

    for directory in [
        agnostic_dir,
        aware_dir,
        agnostic_dir / "visuals",
        aware_dir / "visuals",
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("AquaYOLO TEST-SET ERROR ANALYSIS")
    print("=" * 70)
    print("Dataset :", data_root.resolve())
    print("Weights :", Path(args.weights).resolve())
    print("Device  :", device)

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
        args.weights,
        device,
    )

    model.eval()

    print()
    print("Test images:", len(dataset))
    print()

    modes = {
        "class_agnostic": False,
        "class_aware": True,
    }

    max_visuals_per_mode = 40
    visual_counts = defaultdict(int)

    all_error_rows = defaultdict(list)
    per_class = {}

    confusion_counter = defaultdict(Counter)

    for mode_name in modes:
        per_class[mode_name] = {}

        for class_id, class_name in enumerate(CLASS_NAMES):
            per_class[mode_name][class_id] = {
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth": 0,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "confusion_as_prediction": 0,
                "confused_away": 0,
            }

    total_images = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader):
            images_device = images.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            outputs = model(images_device)

            raw_boxes, raw_scores = decode_raw_outputs(
                outputs
            )

            batch_results = {}

            for mode_name, class_aware in modes.items():
                batch_results[mode_name] = make_predictions(
                    raw_boxes,
                    raw_scores,
                    conf_threshold=args.conf,
                    nms_iou=args.nms_iou,
                    class_aware=class_aware,
                    max_detections=20,
                )

            batch_size = len(targets)

            for local_index in range(batch_size):
                dataset_index = (
                    batch_index * args.batch
                    + local_index
                )

                if dataset_index >= len(dataset):
                    continue

                image_path = Path(
                    dataset.images[dataset_index]
                )

                total_images += 1

                for mode_name in modes:
                    prediction = batch_results[
                        mode_name
                    ][local_index]

                    target = targets[local_index]

                    analysis = analyze_image(
                        prediction,
                        target,
                        iou_threshold=args.match_iou,
                    )

                    for gt_label in analysis["gt_labels"].tolist():
                        per_class[mode_name][gt_label][
                            "ground_truth"
                        ] += 1

                    for item in analysis["tp"]:
                        cid = item["class_id"]

                        per_class[mode_name][cid][
                            "true_positive"
                        ] += 1

                    for item in analysis["fp"]:
                        cid = item["pred_class"]

                        per_class[mode_name][cid][
                            "false_positive"
                        ] += 1

                    for item in analysis["confusions"]:
                        pred_cid = item["pred_class"]
                        gt_cid = item["gt_class"]

                        per_class[mode_name][pred_cid][
                            "confusion_as_prediction"
                        ] += 1

                        per_class[mode_name][gt_cid][
                            "confused_away"
                        ] += 1

                        confusion_counter[mode_name][
                            (
                                CLASS_NAMES[gt_cid],
                                CLASS_NAMES[pred_cid],
                            )
                        ] += 1

                    for item in analysis["fn"]:
                        cid = item["gt_class"]

                        per_class[mode_name][cid][
                            "false_negative"
                        ] += 1

                    event_count = (
                        len(analysis["fp"])
                        + len(analysis["fn"])
                        + len(analysis["confusions"])
                    )

                    if event_count > 0:
                        all_error_rows[mode_name].append(
                            {
                                "image": str(image_path),
                                "false_positives": len(
                                    analysis["fp"]
                                ),
                                "false_negatives": len(
                                    analysis["fn"]
                                ),
                                "confusions": len(
                                    analysis["confusions"]
                                ),
                            }
                        )

                    # Save representative error images.
                    if (
                        event_count > 0
                        and visual_counts[mode_name]
                        < max_visuals_per_mode
                    ):
                        filename = (
                            f"{dataset_index:05d}_"
                            f"error.png"
                        )

                        draw_analysis_image(
                            image_path,
                            analysis,
                            (
                                out_root
                                / mode_name
                                / "visuals"
                                / filename
                            ),
                        )

                        visual_counts[mode_name] += 1

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or (batch_index + 1) == len(loader)
            ):
                print(
                    f"Processed "
                    f"{min((batch_index + 1) * args.batch, len(dataset))}"
                    f"/{len(dataset)} images"
                )

    # --------------------------------------------------------
    # Per-class reports
    # --------------------------------------------------------

    comparison_rows = []

    for mode_name in modes:
        rows = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_confusion = 0

        for class_id in range(NC):
            item = per_class[mode_name][class_id]

            tp = item["true_positive"]
            fp = item["false_positive"]
            fn = item["false_negative"]
            confusion_pred = item[
                "confusion_as_prediction"
            ]
            confusion_away = item[
                "confused_away"
            ]
            gt = item["ground_truth"]

            precision = (
                tp / (tp + fp + confusion_pred)
                if (tp + fp + confusion_pred) > 0
                else 0.0
            )

            recall = (
                tp / gt
                if gt > 0
                else 0.0
            )

            rows.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "ground_truth": gt,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "confusion_as_prediction": confusion_pred,
                    "confused_away": confusion_away,
                    "precision_at_match_iou": precision,
                    "recall_at_match_iou": recall,
                }
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_confusion += confusion_pred

        write_csv(
            out_root
            / mode_name
            / "per_class_errors.csv",
            [
                "class_id",
                "class_name",
                "ground_truth",
                "true_positive",
                "false_positive",
                "false_negative",
                "confusion_as_prediction",
                "confused_away",
                "precision_at_match_iou",
                "recall_at_match_iou",
            ],
            rows,
        )

        combined_precision = (
            total_tp
            / (
                total_tp
                + total_fp
                + total_confusion
            )
            if (
                total_tp
                + total_fp
                + total_confusion
            ) > 0
            else 0.0
        )

        combined_recall = (
            total_tp
            / (
                total_tp
                + total_fn
                + total_confusion
            )
            if (
                total_tp
                + total_fn
                + total_confusion
            ) > 0
            else 0.0
        )

        comparison_rows.append(
            {
                "nms_mode": mode_name,
                "images": total_images,
                "true_positive": total_tp,
                "false_positive": total_fp,
                "false_negative": total_fn,
                "class_confusions": total_confusion,
                "precision_like": combined_precision,
                "recall_like": combined_recall,
            }
        )

        write_csv(
            out_root
            / mode_name
            / "image_errors.csv",
            [
                "image",
                "false_positives",
                "false_negatives",
                "confusions",
            ],
            all_error_rows[mode_name],
        )

        confusion_rows = []

        for (
            gt_name,
            pred_name,
        ), count in confusion_counter[mode_name].most_common():
            confusion_rows.append(
                {
                    "ground_truth_class": gt_name,
                    "predicted_class": pred_name,
                    "count": count,
                }
            )

        write_csv(
            out_root
            / mode_name
            / "confusion_pairs.csv",
            [
                "ground_truth_class",
                "predicted_class",
                "count",
            ],
            confusion_rows,
        )

    write_csv(
        out_root / "nms_comparison.csv",
        [
            "nms_mode",
            "images",
            "true_positive",
            "false_positive",
            "false_negative",
            "class_confusions",
            "precision_like",
            "recall_like",
        ],
        comparison_rows,
    )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    for row in comparison_rows:
        print()
        print("NMS:", row["nms_mode"])
        print(
            "  TP              :",
            row["true_positive"],
        )
        print(
            "  FP              :",
            row["false_positive"],
        )
        print(
            "  FN              :",
            row["false_negative"],
        )
        print(
            "  Class confusions:",
            row["class_confusions"],
        )
        print(
            "  Precision-like  :",
            f"{row['precision_like']:.4f}",
        )
        print(
            "  Recall-like     :",
            f"{row['recall_like']:.4f}",
        )

    print()
    print(
        "Results saved to:",
        out_root.resolve(),
    )


if __name__ == "__main__":
    main()