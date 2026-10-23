from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch
import torchvision
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

NC = 26


def collate_fn(batch):
    images = torch.stack(
        [item[0] for item in batch],
        dim=0,
    )
    targets = [item[1] for item in batch]
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


def decode_predictions_with_score(
    outputs,
    conf_threshold=0.70,
    nms_iou=0.45,
):
    all_boxes = []
    all_scores = []
    all_labels = []

    for output in outputs:
        boxes, scores = decode_scale(output)

        confidence, labels = torch.max(
            scores,
            dim=2,
        )

        boxes = boxes[0]
        confidence = confidence[0]
        labels = labels[0]

        keep = confidence >= conf_threshold

        boxes = boxes[keep]
        confidence = confidence[keep]
        labels = labels[keep]

        if boxes.numel() == 0:
            continue

        boxes = xywh_to_xyxy(boxes)

        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 1)

        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[valid]
        confidence = confidence[valid]
        labels = labels[valid]

        if boxes.numel() == 0:
            continue

        all_boxes.append(boxes)
        all_scores.append(confidence)
        all_labels.append(labels)

    if not all_boxes:
        return (
            torch.empty((0, 4), device=outputs[0]["obj"].device),
            torch.empty((0,), device=outputs[0]["obj"].device),
            torch.empty(
                (0,),
                dtype=torch.long,
                device=outputs[0]["obj"].device,
            ),
        )

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    keep = torchvision.ops.nms(
        boxes,
        scores,
        nms_iou,
    )

    keep = keep[
        torch.argsort(
            scores[keep],
            descending=True,
        )
    ]

    keep = keep[:20]

    return (
        boxes[keep],
        scores[keep],
        labels[keep],
    )


def pairwise_iou(pred_box, gt_boxes):
    if gt_boxes.numel() == 0:
        return torch.empty(
            (0,),
            device=pred_box.device,
        )

    x1 = torch.maximum(
        pred_box[0],
        gt_boxes[:, 0],
    )
    y1 = torch.maximum(
        pred_box[1],
        gt_boxes[:, 1],
    )
    x2 = torch.minimum(
        pred_box[2],
        gt_boxes[:, 2],
    )
    y2 = torch.minimum(
        pred_box[3],
        gt_boxes[:, 3],
    )

    inter = (
        (x2 - x1).clamp(min=0)
        *
        (y2 - y1).clamp(min=0)
    )

    pred_area = (
        (pred_box[2] - pred_box[0]).clamp(min=0)
        *
        (pred_box[3] - pred_box[1]).clamp(min=0)
    )

    gt_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
        *
        (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    )

    union = pred_area + gt_area - inter

    return inter / union.clamp(min=1e-9)


def analyze_image(
    pred_boxes,
    pred_scores,
    pred_labels,
    target,
    iou_threshold=0.50,
):
    target = target.to(pred_boxes.device)
    
    gt_boxes = xywh_to_xyxy(
        target[:, 1:5].float()
    )

    gt_labels = target[:, 0].long()

    matched_gt = set()
    matches = []
    false_positives = []

    order = torch.argsort(
        pred_scores,
        descending=True,
    )

    for pi in order.tolist():

        p_box = pred_boxes[pi]
        p_label = int(pred_labels[pi])
        p_score = float(pred_scores[pi])

        if gt_boxes.numel() == 0:
            false_positives.append(pi)
            continue

        ious = pairwise_iou(
            p_box,
            gt_boxes,
        )

        best_gt = -1
        best_iou = 0.0

        for gi in range(len(gt_boxes)):
            if gi in matched_gt:
                continue

            iou = float(ious[gi])

            if iou >= iou_threshold and iou > best_iou:
                best_gt = gi
                best_iou = iou

        if (
            best_gt >= 0
            and int(gt_labels[best_gt]) == p_label
        ):
            matched_gt.add(best_gt)

            matches.append(
                {
                    "pred_index": pi,
                    "gt_index": best_gt,
                    "class_id": p_label,
                    "score": p_score,
                    "iou": best_iou,
                }
            )
        else:
            false_positives.append(pi)

    return {
        "matches": matches,
        "false_positives": false_positives,
        "matched_gt": matched_gt,
        "gt_count": len(gt_boxes),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=r".\data\AquaYOLO_26class_balanced",
    )
    parser.add_argument(
        "--weights",
        default=r".\runs\aquayolo_fishinv_pretrained_30epoch\best.pt",
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
        default=0.70,
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
        "--top",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(
        r".\runs\aquayolo_fishinv_pretrained_30epoch\demo_gallery"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("AquaYOLO VERIFIED DEMO GALLERY")
    print("=" * 70)

    print("Dataset :", Path(args.data).resolve())
    print("Weights :", Path(args.weights).resolve())
    print("Device  :", device)
    print("Confidence:", args.conf)

    dataset = AquaYOLODataset(
        root=Path(args.data),
        split="val",
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

    candidates = []

    processed = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            outputs = model(images)

            for local_index in range(len(targets)):

                single_outputs = [
                    {
                        key: value[
                            local_index:local_index + 1
                        ]
                        for key, value in output.items()
                    }
                    for output in outputs
                ]

                (
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                ) = decode_predictions_with_score(
                    single_outputs,
                    conf_threshold=args.conf,
                    nms_iou=args.nms_iou,
                )

                analysis = analyze_image(
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    targets[local_index],
                    iou_threshold=args.match_iou,
                )

                if not analysis["matches"]:
                    processed += 1
                    continue

                tp_count = len(
                    analysis["matches"]
                )

                fp_count = len(
                    analysis["false_positives"]
                )

                match_scores = [
                    m["score"]
                    for m in analysis["matches"]
                ]

                mean_score = (
                    sum(match_scores)
                    / len(match_scores)
                )

                mean_iou = (
                    sum(
                        m["iou"]
                        for m in analysis["matches"]
                    )
                    / len(analysis["matches"])
                )

                # Prefer:
                # 1. more correct detections
                # 2. fewer false positives
                # 3. higher IoU
                # 4. higher confidence
                quality = (
                    tp_count * 100.0
                    - fp_count * 15.0
                    + mean_iou * 20.0
                    + mean_score * 10.0
                )

                dataset_index = (
                    batch_index * args.batch
                    + local_index
                )

                candidates.append(
                    {
                        "quality": quality,
                        "dataset_index": dataset_index,
                        "image_path": str(
                            dataset.images[dataset_index]
                        ),
                        "tp": tp_count,
                        "fp": fp_count,
                        "mean_iou": mean_iou,
                        "mean_score": mean_score,
                        "matches": analysis["matches"],
                    }
                )

            processed += len(targets)

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or (batch_index + 1) == len(loader)
            ):
                print(
                    f"Processed "
                    f"{min(processed, len(dataset))}"
                    f"/{len(dataset)} validation images"
                )

    candidates.sort(
        key=lambda item: item["quality"],
        reverse=True,
    )

    selected = []

    # Avoid selecting the exact same class repeatedly.
    class_counts = {}

    for candidate in candidates:

        candidate_classes = {
            m["class_id"]
            for m in candidate["matches"]
        }

        useful = False

        for cid in candidate_classes:
            if class_counts.get(cid, 0) < 2:
                useful = True
                break

        if not useful:
            continue

        selected.append(candidate)

        for cid in candidate_classes:
            class_counts[cid] = (
                class_counts.get(cid, 0) + 1
            )

        if len(selected) >= args.top:
            break

    report_rows = []

    print()
    print("=" * 70)
    print("SELECTED DEMO IMAGES")
    print("=" * 70)

    for rank, item in enumerate(
        selected,
        start=1,
    ):

        source = Path(
            item["image_path"]
        )

        destination = (
            output_dir
            / f"{rank:02d}_{source.name}"
        )

        shutil.copy2(
            source,
            destination,
        )

        predicted_classes = [
            CLASS_NAMES[
                m["class_id"]
            ]
            for m in item["matches"]
        ]

        class_text = ";".join(
            dict.fromkeys(
                predicted_classes
            )
        )

        print(
            f"{rank:02d}. {destination.name}"
        )

        print(
            f"    TP={item['tp']} "
            f"FP={item['fp']} "
            f"mean IoU={item['mean_iou']:.3f} "
            f"mean confidence={item['mean_score']:.3f}"
        )

        print(
            f"    Classes: {class_text}"
        )

        report_rows.append(
            {
                "rank": rank,
                "image": destination.name,
                "source": str(source),
                "true_positives": item["tp"],
                "false_positives": item["fp"],
                "mean_iou": item["mean_iou"],
                "mean_confidence": item["mean_score"],
                "classes": class_text,
            }
        )

    report_path = (
        output_dir
        / "demo_gallery_report.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "image",
                "source",
                "true_positives",
                "false_positives",
                "mean_iou",
                "mean_confidence",
                "classes",
            ],
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print()
    print("=" * 70)
    print("GALLERY COMPLETE")
    print("=" * 70)

    print(
        "Selected images :",
        len(selected),
    )

    print(
        "Gallery folder   :",
        output_dir.resolve(),
    )

    print(
        "Report           :",
        report_path.resolve(),
    )


if __name__ == "__main__":
    main()