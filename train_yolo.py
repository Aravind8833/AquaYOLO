"""AquaYOLO Ultralytics YOLO High-Accuracy Training Pipeline

This script trains a YOLOv8 detector on the AquaYOLO marine species dataset
with transfer learning, mosaic augmentation, and automatic metric graph generation.

Outputs automatically generated in <project>/<name>/:
  - results.png (loss curves + mAP50, mAP50-95, precision, recall)
  - confusion_matrix.png & confusion_matrix_normalized.png
  - PR_curve.png (Precision-Recall curve)
  - F1_curve.png, P_curve.png, R_curve.png
  - val_batch0_labels.jpg, val_batch0_pred.jpg
  - best.pt, last.pt
"""

import argparse
from pathlib import Path
import sys
import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on AquaYOLO Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/AquaYOLO_26class_balanced/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model checkpoint (e.g. yolov8s.pt, FishInv.pt, MegaFauna.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (e.g. 16 for Colab T4 GPU, 4-8 for smaller VRAM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate fraction (cosine lr schedule)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run on (e.g. 0, cpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="./runs/yolo_train",
        help="Output project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="aquayolo_v8_high_acc",
        help="Run name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.pt",
    )
    return parser.parse_args()


def fix_data_yaml_paths(yaml_path: Path) -> Path:
    """Ensure data.yaml uses paths relative to its own location."""
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Set root path to the parent directory of data.yaml
    data["path"] = str(yaml_path.parent.resolve())
    data["train"] = "images/train"
    data["val"] = "images/val"
    data["test"] = "images/test"

    # Save to a resolved temporary yaml to prevent relative path mismatches
    resolved_yaml = yaml_path.parent / "data_resolved.yaml"
    with open(resolved_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return resolved_yaml


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Data config not found at: {data_path}")
        sys.exit(1)

    print("=" * 70)
    print("AQUAYOLO HIGH-ACCURACY YOLO TRAINING")
    print("=" * 70)
    print(f"Data config : {data_path.resolve()}")
    print(f"Base model  : {args.model}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch}")
    print(f"Image size  : {args.imgsz}")
    print(f"Output      : {Path(args.project) / args.name}")
    print("=" * 70)

    # Resolve dataset paths
    resolved_yaml = fix_data_yaml_paths(data_path)

    # Initialize model
    model = YOLO(args.model)

    # Train with optimal marine detection settings
    results = model.train(
        data=str(resolved_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=3,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        device=args.device if args.device else None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        plots=True,
        save=True,
        save_period=10,
    )

    # Evaluate on validation split
    print("\nRunning final validation...")
    val_metrics = model.val(
        data=str(resolved_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=0.6,
        plots=True,
    )

    print("\n" + "=" * 70)
    print("TRAINING & VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Validation mAP@0.50      : {val_metrics.box.map50:.4f}")
    print(f"Validation mAP@0.50:0.95 : {val_metrics.box.map:.4f}")
    print(f"Precision                : {val_metrics.box.mp:.4f}")
    print(f"Recall                   : {val_metrics.box.mr:.4f}")
    print(f"Generated metric plots saved to: {Path(args.project) / args.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
