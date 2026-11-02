import argparse
import csv
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from aquayolo.model import AquaYOLO
from aquayolo.loss import aqua_loss, make_class_weights
from aquayolo.preprocess import enhance_underwater_rgb, resize_image


# ============================================================
# AQUAYOLO DATASET
# ============================================================

CLASS_NAMES = [
    "fish",
    "jellyfish",
    "penguin",
    "puffin",
    "starfish",
    "stingray",
    "shark",
    "crab",
    "lobster",
    "dolphin",
    "whale",
    "turtle",
    "sea_urchin",
    "sea_cucumber",
    "eel",
    "squid",
    "octopus",
    "seal",
    "human",
]

NUM_CLASSES = len(CLASS_NAMES)


def load_class_names(dataset_root):
    """
    Dynamically loads the exact active class names from data.yaml if present,
    ensuring 0-annotation classes are excluded and class IDs match perfectly.
    """
    dataset_root = Path(dataset_root)
    candidates = [
        dataset_root / "data.yaml",
        dataset_root.parent / "data.yaml",
        dataset_root / "dataset.yaml",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                import yaml
                with open(cand, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if "names" in cfg:
                        names = cfg["names"]
                        if isinstance(names, dict):
                            return [names[k] for k in sorted(names.keys())]
                        elif isinstance(names, list):
                            return list(names)
            except Exception:
                pass
    return CLASS_NAMES


class AquaYOLODataset(Dataset):
    """
    YOLO-format dataset.

    Expected directory structure:

        dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        │
        └── labels/
            ├── train/
            ├── val/
            └── test/

    Label format:

        class cx cy width height

    Coordinates are normalized to [0,1].
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        root,
        split,
        size=640,
        augment=False,
        num_classes=None,
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.size = size
        self.augment = augment
        self.num_classes = num_classes or NUM_CLASSES
        self.class_remap = None

        self.image_dir = (
            self.root
            / "images"
            / split
        )

        self.label_dir = (
            self.root
            / "labels"
            / split
        )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}"
            )

        self.images = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in self.IMAGE_EXTENSIONS
            ]
        )

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def remap_classes(self, old_to_new, new_num_classes):
        """Remap class IDs and update active class count."""
        self.num_classes = new_num_classes
        self.class_remap = old_to_new

    def __len__(self):
        return len(self.images)

    def _label_path(self, image_path):
        return (
            self.label_dir
            / f"{image_path.stem}.txt"
        )

    def _read_labels(self, path):
        rows = []

        if not path.exists():
            return rows

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                try:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue

                if self.class_remap is not None:
                    if cls_id not in self.class_remap:
                        continue
                    cls_id = self.class_remap[cls_id]

                if cls_id < 0 or cls_id >= self.num_classes:
                    continue

                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)

                if w <= 0 or h <= 0:
                    continue

                rows.append(
                    [
                        float(cls_id),
                        cx,
                        cy,
                        w,
                        h,
                    ]
                )

        return rows

    @staticmethod
    def _augment_hsv(
        image,
        h_gain=0.015,
        s_gain=0.7,
        v_gain=0.4,
    ):
        """
        Random HSV augmentation for underwater
        color variation robustness.
        """

        r = np.random.uniform(
            -1, 1, 3
        ) * [
            h_gain,
            s_gain,
            v_gain,
        ] + 1

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2HSV
        ).astype(np.float32)

        hsv[..., 0] = (
            hsv[..., 0] * r[0]
        ) % 180

        hsv[..., 1] = np.clip(
            hsv[..., 1] * r[1],
            0,
            255,
        )

        hsv[..., 2] = np.clip(
            hsv[..., 2] * r[2],
            0,
            255,
        )

        return cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2RGB,
        )

    def __getitem__(self, index):

        image_path = self.images[index]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # AquaYOLO physical-domain preprocessing
        # Resize FIRST to target size (640x640) for 10x-30x
        # faster preprocessing on CPU, then apply CLAHE & red boost!
        # ----------------------------------------------------

        image = resize_image(
            image,
            size=self.size
        )

        image = enhance_underwater_rgb(
            image
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        label_path = self._label_path(
            image_path
        )

        labels = self._read_labels(
            label_path
        )

        if len(labels) == 0:
            target = torch.zeros(
                (0, 5),
                dtype=torch.float32
            )
        else:
            target = torch.tensor(
                labels,
                dtype=torch.float32
            )

        # ----------------------------------------------------
        # Data augmentation (training only)
        # ----------------------------------------------------

        if self.augment:

            # Random horizontal flip (50%)
            if random.random() < 0.5:
                image = np.fliplr(
                    image
                ).copy()

                if target.numel() > 0:
                    # Flip cx: cx' = 1 - cx
                    target[:, 1] = (
                        1.0 - target[:, 1]
                    )

            # HSV augmentation
            image = self._augment_hsv(
                image
            )

        # HWC -> CHW
        image_tensor = torch.from_numpy(
            image
        ).float()

        image_tensor = image_tensor.permute(
            2,
            0,
            1
        )

        # [0,255] -> [0,1]
        image_tensor = (
            image_tensor / 255.0
        )

        return image_tensor, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(
        images,
        dim=0
    )

    return images, targets


# ============================================================
# CLASS COUNT
# ============================================================

def calculate_class_counts(dataset, nc=None):
    """
    Count training annotations for class weighting.
    """
    num_c = nc if nc is not None else getattr(dataset, "num_classes", NUM_CLASSES)
    counts = {
        i: 0
        for i in range(num_c)
    }

    remap = getattr(dataset, "class_remap", None)

    for image_path in dataset.images:

        label_path = (
            dataset._label_path(
                image_path
            )
        )

        if not label_path.exists():
            continue

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    cls_id = int(parts[0])
                except ValueError:
                    continue

                if remap is not None:
                    if cls_id not in remap:
                        continue
                    cls_id = remap[cls_id]

                if 0 <= cls_id < num_c:
                    counts[cls_id] += 1

    return counts


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    args,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "image_size": args.size,
        "learning_rate": args.lr,
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    class_weights,
    device,
    epoch,
    total_epochs,
    scaler=None,
    nc=NUM_CLASSES,
):
    model.train()

    running_loss = 0.0
    running_box = 0.0
    running_obj = 0.0
    running_cls = 0.0

    num_batches = len(loader)
    start_time = time.time()
    use_cuda = (device.type == "cuda")

    for step, (images, targets) in enumerate(
        loader,
        start=1
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        targets = [
            t.to(device)
            for t in targets
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward with FP16 Automatic Mixed Precision (AMP)
        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss, loss_items = aqua_loss(
                    outputs=outputs,
                    targets=targets,
                    nc=nc,
                    class_weights=class_weights,
                    box_weight=5.0,
                    obj_weight=1.0,
                    cls_weight=1.0,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_items = aqua_loss(
                outputs=outputs,
                targets=targets,
                nc=nc,
                class_weights=class_weights,
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

        running_loss += float(
            loss.detach().item()
        )

        running_box += (
            loss_items["box"]
        )

        running_obj += (
            loss_items["obj"]
        )

        running_cls += (
            loss_items["cls"]
        )

        # Simple progress display
        if (
            step == 1
            or step % 100 == 0
            or step == num_batches
        ):
            elapsed = (
                time.time()
                -
                start_time
            )

            print(
                f"  Epoch {epoch}/{total_epochs} "
                f"step {step}/{num_batches} "
                f"loss={loss.detach().item():.4f} "
                f"time={elapsed / 60.0:.1f}m"
            )

    n = max(
        num_batches,
        1
    )

    return {
        "loss": running_loss / n,
        "box": running_box / n,
        "obj": running_obj / n,
        "cls": running_cls / n,
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    class_weights,
    device,
    nc=NUM_CLASSES,
):
    model.eval()

    running_loss = 0.0
    running_box = 0.0
    running_obj = 0.0
    running_cls = 0.0

    num_batches = len(loader)
    use_cuda = (device.type == "cuda")

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        targets = [
            t.to(device)
            for t in targets
        ]

        if use_cuda:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss, loss_items = aqua_loss(
                    outputs=outputs,
                    targets=targets,
                    nc=nc,
                    class_weights=class_weights,
                    box_weight=5.0,
                    obj_weight=1.0,
                    cls_weight=1.0,
                )
        else:
            outputs = model(images)
            loss, loss_items = aqua_loss(
                outputs=outputs,
                targets=targets,
                nc=nc,
                class_weights=class_weights,
                box_weight=5.0,
                obj_weight=1.0,
                cls_weight=1.0,
            )

        running_loss += float(
            loss.detach().item()
        )

        running_box += (
            loss_items["box"]
        )

        running_obj += (
            loss_items["obj"]
        )

        running_cls += (
            loss_items["cls"]
        )

    n = max(
        num_batches,
        1
    )

    return {
        "loss": running_loss / n,
        "box": running_box / n,
        "obj": running_obj / n,
        "cls": running_cls / n,
    }


# ============================================================
# HISTORY CSV
# ============================================================

def write_history(
    path,
    history,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_box",
                "train_obj",
                "train_cls",
                "val_loss",
                "val_box",
                "val_obj",
                "val_cls",
            ]
        )

        for row in history:

            writer.writerow(
                [
                    row["epoch"],
                    row["train_loss"],
                    row["train_box"],
                    row["train_obj"],
                    row["train_cls"],
                    row["val_loss"],
                    row["val_box"],
                    row["val_obj"],
                    row["val_cls"],
                ]
            )




# ============================================================
# AUTOMATIC TRAINING GRAPHS
# ============================================================

def plot_history(history, output_dir):
    """Save comprehensive training/validation metric graphs after every epoch."""
    if not history:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed; run: python -m pip install matplotlib")
        return

    epochs = [row["epoch"] for row in history]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall train/validation loss
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train loss")
    ax.plot(epochs, [r["val_loss"] for r in history], marker="o", label="Validation loss")
    ax.set_title("AquaYOLO Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Component losses
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in zip(
        axes,
        ("box", "obj", "cls"),
        ("Box loss", "Objectness loss", "Classification loss"),
    ):
        ax.plot(epochs, [r[f"train_{key}"] for r in history], marker="o", label="Train")
        ax.plot(epochs, [r[f"val_{key}"] for r in history], marker="o", label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_components.png", dpi=150)
    plt.close(fig)

    # mAP / Precision / Recall curves
    has_map = any("mAP50" in r for r in history)
    if has_map:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # mAP50 curve
        axes[0].plot(
            epochs,
            [r.get("mAP50", 0) for r in history],
            marker="o", color="#2196F3", linewidth=2,
        )
        axes[0].set_title("mAP@0.50")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("mAP@0.50")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)

        # Precision curve
        axes[1].plot(
            epochs,
            [r.get("precision", 0) for r in history],
            marker="o", color="#4CAF50", linewidth=2,
        )
        axes[1].set_title("Precision")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Precision")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        # Recall curve
        axes[2].plot(
            epochs,
            [r.get("recall", 0) for r in history],
            marker="o", color="#FF9800", linewidth=2,
        )
        axes[2].set_title("Recall")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Recall")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("AquaYOLO Detection Metrics", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "metrics_curve.png", dpi=150)
        plt.close(fig)

        # Combined results.png (like YOLO standard output)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        metrics = [
            ("train_loss", "Train Loss", "#E53935"),
            ("val_loss", "Val Loss", "#1E88E5"),
            ("mAP50", "mAP@0.50", "#43A047"),
            ("precision", "Precision", "#8E24AA"),
            ("recall", "Recall", "#FB8C00"),
            ("lr", "Learning Rate", "#00ACC1"),
        ]
        for ax, (key, title, color) in zip(axes.flat, metrics):
            vals = [r.get(key, 0) for r in history]
            ax.plot(epochs, vals, marker=".", color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        fig.suptitle("AquaYOLO Training Results", fontsize=16)
        fig.tight_layout()
        fig.savefig(output_dir / "results.png", dpi=150)
        plt.close(fig)

    print(f"Training graphs updated: {output_dir}")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train AquaYOLO"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to AquaYOLO dataset"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32
    )

    parser.add_argument(
        "--size",
        type=int,
        default=640
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3
    )

    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final LR fraction (cosine schedule)"
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Linear warmup epochs"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./runs/aquayolo_custom"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional AquaYOLO initialization checkpoint"
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable training augmentation (enabled by default)"
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentation"
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=300,
        help="Number of validation images to evaluate per intermediate epoch (0 for full)",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    dataset_root = Path(
        args.data
    )

    output_dir = Path(
        args.out
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}"
        )

    if not (
        dataset_root / "data.yaml"
    ).exists():
        raise FileNotFoundError(
            f"data.yaml not found in: {dataset_root}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Load dataset classes dynamically
    # --------------------------------------------------------

    class_names = load_class_names(dataset_root)
    num_classes = len(class_names)

    print()
    print("=" * 70)
    print("AquaYOLO BASELINE TRAINING")
    print("=" * 70)

    print(
        f"Dataset : {dataset_root.resolve()}"
    )

    print(
        f"Classes : {num_classes} active species"
    )

    print(
        f"Device  : {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU     : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Epochs  : {args.epochs}"
    )

    print(
        f"Batch   : {args.batch}"
    )

    print(
        f"Image   : {args.size}"
    )

    print(
        f"LR      : {args.lr}"
    )

    print(
        f"Output  : {output_dir.resolve()}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = AquaYOLODataset(
        root=dataset_root,
        split="train",
        size=args.size,
        augment=(not args.no_augment),
        num_classes=num_classes,
    )

    val_dataset = AquaYOLODataset(
        root=dataset_root,
        split="val",
        size=args.size,
        augment=False,
        num_classes=num_classes,
    )

    print(
        f"Train images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Val images  : "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Class frequencies & Zero-Annotation Filtering
    # --------------------------------------------------------

    print()
    print(
        "Calculating class frequencies..."
    )

    class_counts = calculate_class_counts(
        train_dataset,
        nc=num_classes
    )

    # Automatically filter out any remaining 0-annotation classes from model
    active_indices = [i for i, c in class_counts.items() if c > 0]
    if 0 < len(active_indices) < num_classes:
        zero_classes = [class_names[i] for i in range(num_classes) if class_counts.get(i, 0) == 0]
        print(f"\n⚡ Auto-filtering {len(zero_classes)} zero-annotation species from model head:")
        for zc in zero_classes:
            print(f"   - Excluded '{zc}' (0 annotations)")

        old_to_new = {old_id: new_id for new_id, old_id in enumerate(active_indices)}
        class_names = [class_names[i] for i in active_indices]
        num_classes = len(class_names)

        train_dataset.remap_classes(old_to_new, num_classes)
        val_dataset.remap_classes(old_to_new, num_classes)
        class_counts = {new_id: class_counts[old_id] for old_id, new_id in old_to_new.items()}

    class_weights = make_class_weights(
        class_counts=class_counts,
        nc=num_classes
    ).to(device)

    print()
    print(f"Active training species ({num_classes}):")

    for cls_id, count in class_counts.items():

        cname = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(
            f"  {cls_id:2d} "
            f"{cname:25s} "
            f"{count}"
        )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
        collate_fn=collate_fn,
    )

    # Fast validation subset (e.g. 300 samples) for ~10s metric tracking during intermediate epochs
    if 0 < args.val_samples < len(val_dataset):
        val_eval_dataset = torch.utils.data.Subset(
            val_dataset,
            list(range(min(args.val_samples, len(val_dataset))))
        )
        val_eval_loader = DataLoader(
            val_eval_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.workers > 0),
            collate_fn=collate_fn,
        )
    else:
        val_eval_loader = val_loader

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        f"Creating AquaYOLO model (nc={num_classes})..."
    )

    model = AquaYOLO(
        nc=num_classes
    ).to(device)
    
        # --------------------------------------------------------
    # Optional pretrained initialization
    # --------------------------------------------------------

    if args.pretrained:
        pretrained_path = Path(args.pretrained)

        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: "
                f"{pretrained_path.resolve()}"
            )

        print()
        print(
            "Loading pretrained AquaYOLO initialization:"
        )
        print(
            f"  {pretrained_path.resolve()}"
        )

        checkpoint = torch.load(
            pretrained_path,
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

        print(
            "Pretrained initialization loaded successfully."
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # --------------------------------------------------------
    # Learning rate scheduler
    #   Linear warmup -> Cosine annealing
    # --------------------------------------------------------

    def lr_lambda(epoch_idx):
        """Warmup + Cosine annealing schedule."""
        if epoch_idx < args.warmup_epochs:
            # Linear warmup from 0.1x to 1x
            return 0.1 + 0.9 * (
                epoch_idx / max(args.warmup_epochs, 1)
            )
        # Cosine annealing after warmup
        progress = (
            (epoch_idx - args.warmup_epochs)
            / max(args.epochs - args.warmup_epochs, 1)
        )
        return max(
            args.lrf,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    print(
        f"LR schedule: linear warmup "
        f"{args.warmup_epochs} epochs "
        f"-> cosine annealing to {args.lrf}"
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_mAP50 = 0.0

    best_path = (
        output_dir
        /
        "best.pt"
    )

    history_path = (
        output_dir
        /
        "history.csv"
    )

    # --------------------------------------------------------
    # Mixed precision scaler (FP16 AMP acceleration)
    # --------------------------------------------------------

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        print()
        print(
            "-" * 70
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs}"
            f"  lr={current_lr:.6f}"
        )

        print(
            "-" * 70
        )

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN (with AMP scaler)
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            scaler=scaler,
            nc=num_classes,
        )

        # ----------------------------------------------------
        # VALIDATION (loss)
        # ----------------------------------------------------

        val_metrics = validate(
            model=model,
            loader=val_loader,
            class_weights=class_weights,
            device=device,
            nc=num_classes,
        )

        # ----------------------------------------------------
        # DETECTION METRICS (mAP, precision, recall)
        # ----------------------------------------------------

        mAP50 = 0.0
        precision_val = 0.0
        recall_val = 0.0

        try:
            from aquayolo.evaluate import (
                decode_predictions,
                collect_stats,
                compute_ap,
            )

            model.eval()
            all_preds = []
            all_targets = []

            eval_loader = val_loader if epoch == args.epochs else val_eval_loader

            with torch.no_grad():
                for images, targets in eval_loader:

                    images = images.to(
                        device,
                        non_blocking=True
                    )

                    outputs = model(images)

                    preds = decode_predictions(
                        outputs,
                        conf_threshold=0.25,
                        nms_iou=0.45,
                    )

                    all_preds.extend(preds)
                    all_targets.extend(targets)

            stats = collect_stats(
                all_preds,
                all_targets,
                iou_threshold=0.5,
                num_classes=num_classes,
            )

            aps = []
            total_tp = 0
            total_fp = 0
            total_gt = 0

            for cls_id in range(num_classes):

                cls_stats = stats.get(cls_id, {"num_gt": 0, "scores": [], "matches": []})

                # Skip classes with 0 ground-truth annotations!
                if cls_stats["num_gt"] == 0:
                    continue

                ap = compute_ap(
                    cls_stats["scores"],
                    cls_stats["matches"],
                    cls_stats["num_gt"],
                )

                aps.append(ap)

                total_gt += cls_stats["num_gt"]
                total_tp += sum(cls_stats["matches"])
                total_fp += sum(
                    1 for m in cls_stats["matches"]
                    if not m
                )

            mAP50 = (
                sum(aps) / len(aps)
                if aps
                else 0.0
            )

            precision_val = (
                total_tp / max(total_tp + total_fp, 1)
            )

            recall_val = (
                total_tp / max(total_gt, 1)
            )

        except Exception as e:
            print(
                f"  mAP evaluation warning: {e}"
            )

        # Step the LR scheduler
        scheduler.step()

        epoch_time = (
            time.time()
            -
            epoch_start
        )

        print()

        print(
            f"TRAIN loss="
            f"{train_metrics['loss']:.4f} "
            f"box="
            f"{train_metrics['box']:.4f} "
            f"obj="
            f"{train_metrics['obj']:.4f} "
            f"cls="
            f"{train_metrics['cls']:.4f}"
        )

        print(
            f"VAL   loss="
            f"{val_metrics['loss']:.4f} "
            f"box="
            f"{val_metrics['box']:.4f} "
            f"obj="
            f"{val_metrics['obj']:.4f} "
            f"cls="
            f"{val_metrics['cls']:.4f}"
        )

        print(
            f"EVAL  mAP@0.50={mAP50:.4f} "
            f"precision={precision_val:.4f} "
            f"recall={recall_val:.4f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time / 60.0:.2f} minutes"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history_row = {
            "epoch": epoch,

            "train_loss":
                train_metrics["loss"],

            "train_box":
                train_metrics["box"],

            "train_obj":
                train_metrics["obj"],

            "train_cls":
                train_metrics["cls"],

            "val_loss":
                val_metrics["loss"],

            "val_box":
                val_metrics["box"],

            "val_obj":
                val_metrics["obj"],

            "val_cls":
                val_metrics["cls"],

            "mAP50": mAP50,
            "precision": precision_val,
            "recall": recall_val,
            "lr": current_lr,
        }

        history.append(
            history_row
        )

        write_history(
            history_path,
            history
        )

        # Refresh all graphs after every completed epoch
        plot_history(history, output_dir)

        # ----------------------------------------------------
        # Best model (by mAP@0.5)
        # ----------------------------------------------------

        if mAP50 > best_mAP50:

            best_mAP50 = mAP50

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                args=args,
            )

            print(
                f"NEW BEST MODEL: "
                f"mAP@0.50={best_mAP50:.4f}"
            )

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best checkpoint: "
        f"{best_path}"
    )

    print(
        f"History        : "
        f"{history_path}"
    )

    print(
        f"Best mAP@0.50  : "
        f"{best_mAP50:.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()