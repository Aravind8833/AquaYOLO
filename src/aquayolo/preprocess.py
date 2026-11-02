import cv2
import numpy as np
import torch


# ============================================================
# AQUAYOLO PHYSICAL-DOMAIN PREPROCESSING
# ============================================================

_CLAHE_CACHE = {}


def _get_clahe(clip_limit=2.0, tile_grid_size=(8, 8)):
    key = (clip_limit, tile_grid_size)
    if key not in _CLAHE_CACHE:
        _CLAHE_CACHE[key] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return _CLAHE_CACHE[key]


def enhance_underwater_rgb(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    red_boost=1.20
):
    """
    AquaYOLO physical-domain preprocessing (High-Speed Vectorized).
    """
    if image is None:
        raise ValueError("Input image is None.")

    # 1. RGB -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # 2. L-Channel CLAHE (Cached)
    clahe = _get_clahe(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # 3. LAB -> RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 4. Fast Red-spectrum compensation (in-place slice without copying full image)
    if red_boost != 1.0:
        r_comp = np.clip(enhanced[:, :, 0].astype(np.float32) * red_boost, 0, 255).astype(np.uint8)
        enhanced[:, :, 0] = r_comp

    return enhanced


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_image(
    image,
    size=640
):
    """
    Resize an RGB image to square input size.

    This function performs direct resize to the
    AquaYOLO input resolution.
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    if size <= 0:
        raise ValueError(
            "size must be greater than zero."
        )

    return cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# RGB IMAGE -> MODEL TENSOR
# ============================================================

def image_to_tensor(
    image,
    size=640,
    device=None
):
    """
    Complete preprocessing from RGB NumPy image
    to AquaYOLO model tensor.

    Output shape:

        [1, 3, size, size]

    Pixel range:

        [0, 1]
    """

    enhanced = enhance_underwater_rgb(
        image
    )

    resized = resize_image(
        enhanced,
        size=size
    )

    tensor = torch.from_numpy(
        resized
    ).float()

    # HWC -> CHW
    tensor = tensor.permute(
        2,
        0,
        1
    )

    # uint8 [0,255] -> float [0,1]
    tensor = tensor / 255.0

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    if device is not None:
        tensor = tensor.to(
            device
        )

    return tensor


# ============================================================
# BATCH IMAGE PREPROCESSING
# ============================================================

def preprocess_batch(
    images,
    size=640,
    device=None
):
    """
    Preprocess a list of RGB NumPy images.

    Returns:

        Tensor [B, 3, size, size]
    """

    tensors = []

    for image in images:

        tensor = image_to_tensor(
            image,
            size=size,
            device=None
        )

        tensors.append(
            tensor.squeeze(0)
        )

    if len(tensors) == 0:
        raise ValueError(
            "Cannot preprocess an empty image list."
        )

    batch = torch.stack(
        tensors,
        dim=0
    )

    if device is not None:
        batch = batch.to(
            device
        )

    return batch


# ============================================================
# LOAD IMAGE FROM FILE
# ============================================================

def load_image_rgb(
    image_path
):
    """
    Load an image from disk as RGB.

    OpenCV normally loads BGR, so it is explicitly
    converted to RGB here.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# LOAD + PREPROCESS
# ============================================================

def load_and_preprocess(
    image_path,
    size=640,
    device=None
):
    """
    Load an image from disk and prepare it
    for AquaYOLO inference/training.
    """

    image = load_image_rgb(
        image_path
    )

    return image_to_tensor(
        image,
        size=size,
        device=device
    )