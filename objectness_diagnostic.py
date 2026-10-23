from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aquayolo.model import AquaYOLO
from aquayolo.train import AquaYOLODataset
from aquayolo.loss import build_targets


DATASET = Path(
    r".\data\AquaYOLO_26class_balanced"
)

WEIGHTS = Path(
    r".\runs\aquayolo_fishinv_pretrained_30epoch\best.pt"
)

IMAGE_SIZE = 640
NC = 26
BATCH_SIZE = 4
WORKERS = 2


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


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("AquaYOLO OBJECTNESS DIAGNOSTIC")
    print("=" * 70)

    print(
        "Dataset :",
        DATASET.resolve(),
    )

    print(
        "Weights :",
        WEIGHTS.resolve(),
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
        root=DATASET,
        split="val",
        size=IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = AquaYOLO(
        nc=NC
    ).to(device)

    load_checkpoint(
        model,
        WEIGHTS,
        device,
    )

    model.eval()

    scales = {
        0: "P3",
        1: "P4",
        2: "P5",
    }

    statistics = {}

    for scale_index in range(3):

        statistics[scale_index] = {
            "positive": [],
            "negative_sample": [],
            "negative_above_01": 0,
            "negative_above_03": 0,
            "negative_above_05": 0,
            "negative_above_08": 0,
            "negative_total": 0,
            "positive_total": 0,
            "bce_positive": 0.0,
            "bce_negative_sample": 0.0,
        }

    total_images = 0

    with torch.no_grad():

        for batch_index, (images, targets) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=torch.cuda.is_available(),
            )

            outputs = model(images)

            grid_sizes = [
                (
                    outputs[i]["obj"].shape[-2],
                    outputs[i]["obj"].shape[-1],
                )
                for i in range(3)
            ]

            target_scales = build_targets(
                targets=targets,
                batch_size=images.shape[0],
                nc=NC,
                grid_sizes=grid_sizes,
            )

            for scale_index in range(3):

                logits = outputs[
                    scale_index
                ]["obj"]

                probability = torch.sigmoid(
                    logits
                )

                (
                    obj_target,
                    _,
                    _,
                    _,
                ) = target_scales[
                    scale_index
                ]

                positive_mask = (
                    obj_target > 0.5
                )

                negative_mask = (
                    obj_target <= 0.5
                )

                pos_prob = probability[
                    positive_mask
                ]

                neg_prob = probability[
                    negative_mask
                ]

                stats = statistics[
                    scale_index
                ]

                if pos_prob.numel() > 0:

                    stats[
                        "positive"
                    ].append(
                        pos_prob.cpu()
                    )

                    stats[
                        "positive_total"
                    ] += pos_prob.numel()

                    pos_bce = torch.nn.functional.binary_cross_entropy(
                        pos_prob,
                        torch.ones_like(
                            pos_prob
                        ),
                        reduction="sum",
                    )

                    stats[
                        "bce_positive"
                    ] += float(
                        pos_bce.item()
                    )

                if neg_prob.numel() > 0:

                    stats[
                        "negative_sample"
                    ].append(
                        neg_prob.cpu()
                    )

                    stats[
                        "negative_total"
                    ] += neg_prob.numel()

                    stats[
                        "negative_above_01"
                    ] += int(
                        (neg_prob > 0.10).sum().item()
                    )

                    stats[
                        "negative_above_03"
                    ] += int(
                        (neg_prob > 0.30).sum().item()
                    )

                    stats[
                        "negative_above_05"
                    ] += int(
                        (neg_prob > 0.50).sum().item()
                    )

                    stats[
                        "negative_above_08"
                    ] += int(
                        (neg_prob > 0.80).sum().item()
                    )

                    # Sample the top 10% exactly as
                    # the current loss does.
                    total_negative = neg_prob.numel()

                    hard_k = max(
                        64,
                        int(
                            0.10
                            * total_negative
                        ),
                    )

                    hard_k = min(
                        hard_k,
                        total_negative,
                    )

                    _, indices = torch.topk(
                        neg_prob,
                        k=hard_k,
                        largest=True,
                        sorted=False,
                    )

                    hard_prob = neg_prob[
                        indices
                    ]

                    hard_bce = torch.nn.functional.binary_cross_entropy(
                        hard_prob,
                        torch.zeros_like(
                            hard_prob
                        ),
                        reduction="sum",
                    )

                    stats[
                        "bce_negative_sample"
                    ] += float(
                        hard_bce.item()
                    )

            total_images += images.shape[0]

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

    print()
    print("=" * 70)
    print("OBJECTNESS RESULTS")
    print("=" * 70)

    for scale_index in range(3):

        stats = statistics[
            scale_index
        ]

        positives = torch.cat(
            stats["positive"]
        ) if stats["positive"] else torch.empty(0)

        negative_samples = torch.cat(
            stats["negative_sample"]
        ) if stats["negative_sample"] else torch.empty(0)

        positive_mean = (
            positives.mean().item()
            if positives.numel()
            else 0.0
        )

        positive_median = (
            positives.median().item()
            if positives.numel()
            else 0.0
        )

        negative_mean = (
            negative_samples.mean().item()
            if negative_samples.numel()
            else 0.0
        )

        negative_max = (
            negative_samples.max().item()
            if negative_samples.numel()
            else 0.0
        )

        total_negative = stats[
            "negative_total"
        ]

        print()
        print(
            f"{scales[scale_index]}"
        )
        print("-" * 40)

        print(
            "Positive cells      :",
            stats["positive_total"],
        )

        print(
            "Positive mean prob  :",
            f"{positive_mean:.4f}",
        )

        print(
            "Positive median     :",
            f"{positive_median:.4f}",
        )

        print(
            "Background cells    :",
            total_negative,
        )

        print(
            "Background mean*    :",
            f"{negative_mean:.4f}",
        )

        print(
            "Background max*     :",
            f"{negative_max:.4f}",
        )

        if total_negative > 0:

            print(
                "Background > 0.10   :",
                f"{100.0 * stats['negative_above_01'] / total_negative:.3f}%",
            )

            print(
                "Background > 0.30   :",
                f"{100.0 * stats['negative_above_03'] / total_negative:.3f}%",
            )

            print(
                "Background > 0.50   :",
                f"{100.0 * stats['negative_above_05'] / total_negative:.3f}%",
            )

            print(
                "Background > 0.80   :",
                f"{100.0 * stats['negative_above_08'] / total_negative:.3f}%",
            )

    print()
    print("* Background statistics are sampled from the current")
    print("  hard-negative mining strategy (top 10%).")
    print()
    print(
        "Diagnostic complete."
    )


if __name__ == "__main__":
    main()