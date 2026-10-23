import cv2
import numpy as np
import torch


# ============================================================
# AQUAYOLO PHYSICAL-DOMAIN PREPROCESSING
# ============================================================

def enhance_underwater_rgb(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    red_boost=1.20
):
    """
    AquaYOLO physical-domain preprocessing.

    Pipeline:

        Input RGB image
              |
              v
        RGB -> LAB
              |
              v
        L-channel CLAHE
              |
              v
        Red-spectrum boost
              |
              v
        Enhanced RGB image

    Parameters
    ----------
    image:
        NumPy image in RGB format, uint8,
        shape [H, W, 3].

    clip_limit:
        CLAHE contrast limit.

    tile_grid_size:
        CLAHE tile grid.

    red_boost:
        Multiplicative red-channel compensation factor.

    Returns
    -------
    enhanced_rgb:
        uint8 RGB image.
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    if not isinstance(
        image,
        np.ndarray
    ):
        raise TypeError(
            "Input image must be a NumPy array."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected image shape [H, W, 3]."
        )

    if image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    # ========================================================
    # STEP 1: RGB -> LAB
    # ========================================================

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    # ========================================================
    # STEP 2: L-CHANNEL CLAHE
    # ========================================================

    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    lab[:, :, 0] = clahe.apply(
        l_channel
    )

    # LAB -> RGB
    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # ========================================================
    # STEP 3: RED-SPECTRUM COMPENSATION
    # ========================================================

    enhanced = enhanced.astype(
        np.float32
    )

    enhanced[:, :, 0] *= red_boost

    enhanced[:, :, 0] = np.clip(
        enhanced[:, :, 0],
        0,
        255
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

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