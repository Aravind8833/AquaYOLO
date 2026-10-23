from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aquayolo.model import AquaYOLO
from aquayolo.train import AquaYOLODataset

from error_analysis import (
    NC,
    collate_fn,
    load_checkpoint,
    decode_raw_outputs,
    make_predictions,
    analyze_image,
)


THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument(
        "--out",
        default=(
            ".\\runs\\"
            "aquayolo_fishinv_pretrained_30epoch\\"
            "error_analysis\\confidence_sweep.csv"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = AquaYOLODataset(
        root=Path(args.data),
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

    model = AquaYOLO(nc=NC).to(device)

    load_checkpoint(
        model,
        args.weights,
        device,
    )

    model.eval()

    modes = {
        "class_agnostic": False,
        "class_aware": True,
    }

    totals = {}

    for mode_name in modes:
        for threshold in THRESHOLDS:
            key = (
                mode_name,
                threshold,
            )

            totals[key] = {
                "images": 0,
                "predictions": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "confusions": 0,
            }

    print("=" * 70)
    print("AquaYOLO CONFIDENCE THRESHOLD SWEEP")
    print("=" * 70)

    with torch.no_grad():

        for batch_index, (images, targets) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            outputs = model(images)

            raw_boxes, raw_scores = decode_raw_outputs(
                outputs
            )

            for threshold in THRESHOLDS:

                for mode_name, class_aware in modes.items():

                    predictions = make_predictions(
                        raw_boxes,
                        raw_scores,
                        conf_threshold=threshold,
                        nms_iou=args.nms_iou,
                        class_aware=class_aware,
                        max_detections=20,
                    )

                    key = (
                        mode_name,
                        threshold,
                    )

                    for local_index, target in enumerate(targets):

                        prediction = predictions[
                            local_index
                        ]

                        analysis = analyze_image(
                            prediction,
                            target,
                            iou_threshold=args.match_iou,
                        )

                        totals[key]["images"] += 1

                        totals[key]["predictions"] += len(
                            prediction["scores"]
                        )

                        totals[key]["tp"] += len(
                            analysis["tp"]
                        )

                        totals[key]["fp"] += len(
                            analysis["fp"]
                        )

                        totals[key]["fn"] += len(
                            analysis["fn"]
                        )

                        totals[key]["confusions"] += len(
                            analysis["confusions"]
                        )

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or (batch_index + 1) == len(loader)
            ):
                processed = min(
                    (batch_index + 1) * args.batch,
                    len(dataset),
                )

                print(
                    f"Processed {processed}/{len(dataset)}"
                )

    rows = []

    for mode_name in modes:
        for threshold in THRESHOLDS:

            values = totals[
                (
                    mode_name,
                    threshold,
                )
            ]

            tp = values["tp"]
            fp = values["fp"]
            fn = values["fn"]
            confusions = values["confusions"]
            images = values["images"]
            predictions = values["predictions"]

            precision = (
                tp / (tp + fp + confusions)
                if (tp + fp + confusions) > 0
                else 0.0
            )

            recall = (
                tp / (
                    tp
                    + fn
                    + confusions
                )
                if (
                    tp
                    + fn
                    + confusions
                ) > 0
                else 0.0
            )

            predictions_per_image = (
                predictions / images
                if images > 0
                else 0.0
            )

            rows.append(
                {
                    "nms_mode": mode_name,
                    "confidence_threshold": threshold,
                    "images": images,
                    "predictions": predictions,
                    "predictions_per_image":
                        predictions_per_image,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "confusions": confusions,
                    "precision_like": precision,
                    "recall_like": recall,
                }
            )

    out_path = Path(args.out)

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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for row in rows:

        print(
            f"{row['nms_mode']:15s} "
            f"conf={row['confidence_threshold']:.2f} "
            f"pred/img={row['predictions_per_image']:.2f} "
            f"TP={row['tp']:4d} "
            f"FP={row['fp']:5d} "
            f"FN={row['fn']:4d} "
            f"conf={row['confusions']:4d} "
            f"P={row['precision_like']:.4f} "
            f"R={row['recall_like']:.4f}"
        )

    print()
    print("Saved:", out_path.resolve())


if __name__ == "__main__":
    main()