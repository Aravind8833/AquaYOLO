"""
AquaYOLO EigenCAM Visual Explanations Module
===========================================

Visualizes the spatial and spectral attention of the AquaYOLO model:
- SAC: Spectral Attenuation Compensator activation
- TCG: Turbidity Contrast Gate spatial focus
- Aqua-Attn: Combined feature response
- Bounding Box overlay of detected marine organisms

Based on principal component analysis (PCA) of feature representations.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from aquayolo.model import AquaYOLO
from aquayolo.preprocess import enhance_underwater_rgb, resize_image
from aquayolo.train import CLASS_NAMES, NUM_CLASSES


class ActivationCapture:
    """Forward hook container to capture layer activations during inference."""

    def __init__(self):
        self.activation = None

    def __call__(self, module, inputs, output):
        if torch.is_tensor(output):
            self.activation = output.detach().float().cpu()


def compute_pca_heatmap(activation, output_size=(640, 640)):
    """
    Computes an EigenCAM heatmap using the 1st principal component of features.
    If single-channel (like TCG spatial gate), normalizes directly.
    """
    if activation is None:
        return None

    if activation.ndim != 4:
        return None

    features = activation[0]  # [C, H, W]
    channels, height, width = features.shape

    if channels == 1:
        heat = features[0].numpy()
    else:
        # Reshape to [H*W, C]
        features_flat = features.permute(1, 2, 0).reshape(height * width, channels)
        features_flat = features_flat - features_flat.mean(dim=0, keepdim=True)

        try:
            U, S, _ = torch.pca_lowrank(features_flat, q=1, center=False)
            heat = (U[:, 0] * S[0]).abs().numpy()
        except RuntimeError:
            heat = features_flat.abs().mean(dim=1).numpy()

        heat = heat.reshape(height, width)

    # Normalize to [0, 1]
    heat = heat - heat.min()
    denom = heat.max() + 1e-8
    heat = heat / denom

    # Resize to target image dimensions
    heat = cv2.resize(heat, output_size, interpolation=cv2.INTER_CUBIC)
    return np.clip(heat, 0.0, 1.0)


def overlay_heatmap_on_image(image_rgb, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlays a normalized heatmap onto an RGB image."""
    if heatmap is None:
        return image_rgb

    heat_uint8 = (heatmap * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, colored_rgb, alpha, 0)
    return blended


def explain_image(model, device, image_path, conf_thresh=0.25, output_dir=None):
    """
    Runs inference on a single underwater image, capturing SAC, TCG, and Aqua-Attn
    activations, and generates side-by-side EigenCAM explanation figures.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not load image at {image_path}")

    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_rgb.shape[:2]

    # Preprocessing: LAB CLAHE + Red spectrum compensation
    enhanced_rgb = enhance_underwater_rgb(orig_rgb)
    resized_rgb = resize_image(enhanced_rgb, 640)

    tensor = torch.from_numpy(resized_rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # Register forward hooks on Aqua-Attn modules (SAC, TCG gate, and overall AquaAttn)
    sac_p5_cap = ActivationCapture()
    tcg_p5_cap = ActivationCapture()
    attn_p5_cap = ActivationCapture()

    sac_hook = model.aqua_attn_p5.sac.register_forward_hook(sac_p5_cap)
    tcg_hook = model.aqua_attn_p5.tcg.gate.register_forward_hook(tcg_p5_cap)
    attn_hook = model.aqua_attn_p5.register_forward_hook(attn_p5_cap)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)

    # Remove hooks
    sac_hook.remove()
    tcg_hook.remove()
    attn_hook.remove()

    # Generate heatmaps
    out_shape = (resized_rgb.shape[1], resized_rgb.shape[0])
    sac_heat = compute_pca_heatmap(sac_p5_cap.activation, out_shape)
    tcg_heat = compute_pca_heatmap(tcg_p5_cap.activation, out_shape)
    attn_heat = compute_pca_heatmap(attn_p5_cap.activation, out_shape)

    sac_overlay = overlay_heatmap_on_image(resized_rgb, sac_heat, cv2.COLORMAP_TURBO, alpha=0.45)
    tcg_overlay = overlay_heatmap_on_image(resized_rgb, tcg_heat, cv2.COLORMAP_HOT, alpha=0.50)
    attn_overlay = overlay_heatmap_on_image(resized_rgb, attn_heat, cv2.COLORMAP_JET, alpha=0.45)

    # Plot comprehensive explanation figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor("#111827")

    titles = [
        "1. Raw Underwater Image",
        "2. LAB CLAHE + Red Comp",
        "3. SAC Spectral Attention",
        "4. TCG Turbidity Gate",
        "5. Aqua-Attn (EigenCAM)",
    ]
    images = [
        cv2.resize(orig_rgb, (640, 640)),
        resized_rgb,
        sac_overlay,
        tcg_overlay,
        attn_overlay,
    ]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout()

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_path = out_p / f"eigencam_{Path(image_path).stem}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved EigenCAM visualization: {save_path}")
        return save_path

    return fig


def main():
    parser = argparse.ArgumentParser(description="AquaYOLO EigenCAM Visual Explanations")
    parser.add_argument("--weights", type=str, required=True, help="Path to AquaYOLO .pt checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--out", type=str, default="./runs/eigencam", help="Output directory")
    parser.add_argument("--device", type=str, default="", help="Device: cuda or cpu")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

    if "head.cls_p3.weight" in state_dict:
        model_nc = state_dict["head.cls_p3.weight"].shape[0]
    elif isinstance(ckpt, dict) and "nc" in ckpt:
        model_nc = ckpt["nc"]
    else:
        model_nc = NUM_CLASSES

    print(f"Loading AquaYOLO ({model_nc} classes) weights from {args.weights} on {device}...")
    model = AquaYOLO(nc=model_nc).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    img_path = Path(args.image)
    if img_path.is_dir():
        imgs = sorted(list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")))[:10]
    else:
        imgs = [img_path]

    for p in imgs:
        explain_image(model, device, p, output_dir=args.out)


if __name__ == "__main__":
    main()
