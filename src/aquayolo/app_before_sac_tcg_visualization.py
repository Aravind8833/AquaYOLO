import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

from aquayolo.model import AquaYOLO
from aquayolo.preprocess import enhance_underwater_rgb, resize_image
from aquayolo.evaluate import decode_predictions
from aquayolo.train import CLASS_NAMES, NUM_CLASSES


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AquaYOLO",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "aquayolo_fishinv_pretrained_30epoch"
    / "best.pt"
)

IMAGE_SIZE = 640


# ============================================================
# MARINE BIOLOGICAL METADATA
# ============================================================

SPECIES_METADATA = {

    "jellyfish": {
        "common_name": "Jellyfish",
        "scientific_name": "Cnidaria",
        "taxonomy": "Phylum Cnidaria",
        "habitat": "Marine waters, coastal zones and open ocean",
        "toxicity": "Some species have venomous stings; toxicity varies considerably by species.",
        "edibility": "Some species are consumed in certain cuisines, but species identification is important.",
        "description": "Soft-bodied marine animals characterized by an umbrella-shaped bell and tentacles."
    },

    "penguin": {
        "common_name": "Penguin",
        "scientific_name": "Sphenisciformes",
        "taxonomy": "Order Sphenisciformes",
        "habitat": "Southern Hemisphere marine and coastal environments",
        "toxicity": "Not generally considered venomous.",
        "edibility": "Not treated as a normal food species.",
        "description": "Flightless seabirds adapted for swimming and underwater hunting."
    },

    "puffin": {
        "common_name": "Puffin",
        "scientific_name": "Fratercula",
        "taxonomy": "Genus Fratercula",
        "habitat": "Cold northern marine and coastal regions",
        "toxicity": "Not venomous.",
        "edibility": "Historically consumed in some regions; not a general food recommendation.",
        "description": "Seabirds recognized by their compact bodies and distinctive colorful bills."
    },

    "starfish": {
        "common_name": "Starfish / Sea Star",
        "scientific_name": "Asteroidea",
        "taxonomy": "Class Asteroidea",
        "habitat": "Rocky shores, reefs and marine seabeds",
        "toxicity": "Most species are not dangerous to humans.",
        "edibility": "Generally not considered a common food species.",
        "description": "Marine echinoderms commonly having radial symmetry and multiple arms."
    },

    "stingray": {
        "common_name": "Stingray",
        "scientific_name": "Batoidea",
        "taxonomy": "Superorder Batoidea",
        "habitat": "Coastal marine environments and sandy or muddy seafloors",
        "toxicity": "Some species possess venomous tail spines.",
        "edibility": "Some species are eaten in various regions.",
        "description": "Flattened cartilaginous fishes with broad pectoral fins."
    },

    "basking_shark": {
        "common_name": "Basking Shark",
        "scientific_name": "Cetorhinus maximus",
        "taxonomy": "Family Cetorhinidae",
        "habitat": "Temperate coastal and offshore waters",
        "toxicity": "Not venomous.",
        "edibility": "Historically utilized commercially in some regions; conservation considerations apply.",
        "description": "A large filter-feeding shark that primarily consumes plankton."
    },

    "blacktip_reef_shark": {
        "common_name": "Blacktip Reef Shark",
        "scientific_name": "Carcharhinus melanopterus",
        "taxonomy": "Family Carcharhinidae",
        "habitat": "Tropical coral reefs and shallow coastal waters",
        "toxicity": "Not venomous.",
        "edibility": "Shark meat is consumed in some regions.",
        "description": "A reef-associated shark commonly recognized by dark fin markings."
    },

    "great_white_shark": {
        "common_name": "Great White Shark",
        "scientific_name": "Carcharodon carcharias",
        "taxonomy": "Family Lamnidae",
        "habitat": "Temperate and subtropical marine waters",
        "toxicity": "Not venomous.",
        "edibility": "Not considered a standard food recommendation.",
        "description": "Large predatory shark with a characteristic conical snout."
    },

    "hammerhead_shark": {
        "common_name": "Hammerhead Shark",
        "scientific_name": "Sphyrnidae",
        "taxonomy": "Family Sphyrnidae",
        "habitat": "Tropical and temperate coastal waters",
        "toxicity": "Not venomous.",
        "edibility": "Some shark species are commercially utilized, but conservation status varies.",
        "description": "Sharks characterized by a distinctive laterally expanded head."
    },

    "nurse_shark": {
        "common_name": "Nurse Shark",
        "scientific_name": "Ginglymostoma cirratum",
        "taxonomy": "Family Ginglymostomatidae",
        "habitat": "Warm shallow coastal waters and reef environments",
        "toxicity": "Not venomous.",
        "edibility": "Shark meat may be consumed in some regions.",
        "description": "A bottom-associated shark often resting on sandy or rocky seabeds."
    },

    "oceanic_whitetip_shark": {
        "common_name": "Oceanic Whitetip Shark",
        "scientific_name": "Carcharhinus longimanus",
        "taxonomy": "Family Carcharhinidae",
        "habitat": "Open-ocean tropical and subtropical waters",
        "toxicity": "Not venomous.",
        "edibility": "Shark meat is used in some fisheries.",
        "description": "An oceanic shark distinguished by broad fins with pale or white markings."
    },

    "thresher_shark": {
        "common_name": "Thresher Shark",
        "scientific_name": "Alopias",
        "taxonomy": "Family Alopiidae",
        "habitat": "Open ocean and coastal waters",
        "toxicity": "Not venomous.",
        "edibility": "Some fisheries utilize thresher sharks.",
        "description": "Recognized by an extremely elongated upper tail lobe."
    },

    "whale_shark": {
        "common_name": "Whale Shark",
        "scientific_name": "Rhincodon typus",
        "taxonomy": "Family Rhincodontidae",
        "habitat": "Warm tropical and subtropical marine waters",
        "toxicity": "Not venomous.",
        "edibility": "Protected in many regions and not a normal food species.",
        "description": "The world's largest known fish and a filter feeder."
    },

    "whitetip_reef_shark": {
        "common_name": "Whitetip Reef Shark",
        "scientific_name": "Triaenodon obesus",
        "taxonomy": "Family Carcharhinidae",
        "habitat": "Tropical coral reefs and rocky marine habitats",
        "toxicity": "Not venomous.",
        "edibility": "Shark meat is consumed in some locations.",
        "description": "A reef-dwelling shark with distinctive white-tipped fins."
    },

    "ballan_wrasse": {
        "common_name": "Ballan Wrasse",
        "scientific_name": "Labrus bergylta",
        "taxonomy": "Family Labridae",
        "habitat": "Rocky coastal habitats of the northeastern Atlantic",
        "toxicity": "Not generally considered venomous.",
        "edibility": "Edible; also used in some marine aquaculture contexts.",
        "description": "A large wrasse associated with rocky Atlantic coastal habitats."
    },

    "cuckoo_wrasse": {
        "common_name": "Cuckoo Wrasse",
        "scientific_name": "Labrus mixtus",
        "taxonomy": "Family Labridae",
        "habitat": "Rocky Atlantic and European coastal waters",
        "toxicity": "Not generally venomous.",
        "edibility": "Edibility is reported, although it is not a major food fish.",
        "description": "A colorful Atlantic wrasse with strong sexual dimorphism."
    },

    "corkwing_wrasse": {
        "common_name": "Corkwing Wrasse",
        "scientific_name": "Symphodus melops",
        "taxonomy": "Family Labridae",
        "habitat": "Rocky coastal waters of the northeastern Atlantic",
        "toxicity": "Not generally venomous.",
        "edibility": "Can be consumed locally.",
        "description": "A coastal wrasse commonly found around reefs and rocky habitats."
    },

    "goldsinny_wrasse": {
        "common_name": "Goldsinny Wrasse",
        "scientific_name": "Ctenolabrus rupestris",
        "taxonomy": "Family Labridae",
        "habitat": "Rocky Atlantic coastal waters",
        "toxicity": "Not generally venomous.",
        "edibility": "Not a major commercial food species.",
        "description": "A small wrasse associated with rocky seabeds."
    },

    "rock_cook": {
        "common_name": "Rock Cook",
        "scientific_name": "Centrolabrus exoletus",
        "taxonomy": "Family Labridae",
        "habitat": "Rocky northeastern Atlantic coast",
        "toxicity": "Not generally venomous.",
        "edibility": "Not generally a major food species.",
        "description": "A small Atlantic wrasse found around rocky habitats."
    },

    "atlantic_cod": {
        "common_name": "Atlantic Cod",
        "scientific_name": "Gadus morhua",
        "taxonomy": "Family Gadidae",
        "habitat": "North Atlantic marine waters",
        "toxicity": "Not venomous.",
        "edibility": "Widely recognized as an important food fish.",
        "description": "A commercially important North Atlantic fish."
    },

    "pollack": {
        "common_name": "Pollack",
        "scientific_name": "Pollachius pollachius",
        "taxonomy": "Family Gadidae",
        "habitat": "Northeastern Atlantic coastal and offshore waters",
        "toxicity": "Not venomous.",
        "edibility": "Edible food fish.",
        "description": "A predatory gadid fish found near rocky coastal areas."
    },

    "whiting": {
        "common_name": "Whiting",
        "scientific_name": "Merlangius merlangus",
        "taxonomy": "Family Gadidae",
        "habitat": "North Atlantic and European coastal waters",
        "toxicity": "Not venomous.",
        "edibility": "Commonly eaten food fish.",
        "description": "A slender gadid fish found over sandy and muddy seabeds."
    },

    "spiny_dogfish": {
        "common_name": "Spiny Dogfish",
        "scientific_name": "Squalus acanthias",
        "taxonomy": "Family Squalidae",
        "habitat": "Temperate coastal and offshore marine waters",
        "toxicity": "Defensive dorsal spines can cause injury; not typically described as venomous.",
        "edibility": "Edible and used in some fisheries.",
        "description": "A small shark-like fish with characteristic dorsal spines."
    },

    "brown_crab": {
        "common_name": "Brown Crab",
        "scientific_name": "Cancer pagurus",
        "taxonomy": "Family Cancridae",
        "habitat": "Rocky and sandy North Atlantic seabeds",
        "toxicity": "Not venomous.",
        "edibility": "Important edible crab species.",
        "description": "A large edible crab found in northeastern Atlantic coastal waters."
    },

    "two_spot_goby": {
        "common_name": "Two Spot Goby",
        "scientific_name": "Pomatoschistus flavescens",
        "taxonomy": "Family Gobiidae",
        "habitat": "Shallow northeastern Atlantic coastal waters",
        "toxicity": "Not generally venomous.",
        "edibility": "Not a major food species.",
        "description": "A small benthic goby associated with coastal seabeds."
    },

    "human": {
        "common_name": "Human",
        "scientific_name": "Homo sapiens",
        "taxonomy": "Species Homo sapiens",
        "habitat": "Artificially introduced into marine scene; not a marine species.",
        "toxicity": "Not applicable.",
        "edibility": "Not applicable.",
        "description": "Human detected in an underwater image or diving environment."
    },
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(weights_path):
    """
    Load AquaYOLO checkpoint once.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AquaYOLO(
        nc=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model, device


# ============================================================
# EIGENCAM-STYLE ACTIVATION CAPTURE
# ============================================================

class ActivationCapture:
    def __init__(self):
        self.activation = None

    def __call__(
        self,
        module,
        inputs,
        output
    ):
        self.activation = (
            output.detach()
            .float()
            .cpu()
        )


def make_aqua_attention_heatmap(
    activation,
    output_size,
):
    """
    Create an EigenCAM-style heatmap from the
    Aqua-Attn output activation.

    This is an explanatory visualization of
    feature activation; it should not be interpreted
    as a proof of biological reasoning.
    """

    if activation is None:
        return None

    # Expected:
    # [B,C,H,W]
    if activation.ndim != 4:
        return None

    features = activation[0]

    channels = features.shape[0]
    height = features.shape[1]
    width = features.shape[2]

    # C,H,W -> H,W,C
    features = (
        features
        .permute(1, 2, 0)
        .contiguous()
    )

    features = features.reshape(
        height * width,
        channels,
    )

    features = (
        features
        -
        features.mean(
            dim=0,
            keepdim=True
        )
    )

    # PCA / first principal component.
    try:

        U, S, V = torch.pca_lowrank(
            features,
            q=1,
            center=False,
        )

        heat = (
            U[:, 0]
            *
            S[0]
        ).abs()

    except RuntimeError:

        # Fallback: mean absolute activation.
        heat = features.abs().mean(
            dim=1
        )

    heat = heat.reshape(
        height,
        width
    ).numpy()

    heat -= heat.min()

    denom = (
        heat.max()
        +
        1e-8
    )

    heat /= denom

    heat = cv2.resize(
        heat,
        output_size,
        interpolation=cv2.INTER_CUBIC,
    )

    return heat


# ============================================================
# IMAGE INFERENCE
# ============================================================

def run_inference(
    model,
    device,
    image_rgb,
    confidence,
    nms_iou,
):
    """
    Preprocess and infer one image.
    """

    enhanced = enhance_underwater_rgb(
        image_rgb
    )

    resized = resize_image(
        enhanced,
        IMAGE_SIZE
    )

    tensor = torch.from_numpy(
        resized
    ).float()

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor / 255.0

    tensor = tensor.unsqueeze(
        0
    ).to(
        device
    )

    # --------------------------------------------------------
    # Capture Aqua-Attn output.
    # --------------------------------------------------------

    capture = ActivationCapture()

    hook = model.aqua_attn.register_forward_hook(
        capture
    )

    try:

        with torch.no_grad():
            outputs = model(
                tensor
            )

    finally:

        hook.remove()

    predictions = decode_predictions(
        outputs,
        conf_threshold=confidence,
        nms_iou=nms_iou,
    )[0]

    return (
        resized,
        predictions,
        capture.activation,
    )


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image_rgb,
    predictions,
):
    """
    Draw detection boxes.
    """

    image = Image.fromarray(
        image_rgb
    ).convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        image
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            18
        )
    except Exception:
        font = ImageFont.load_default()

    for box, score, label in zip(
        predictions["boxes"],
        predictions["scores"],
        predictions["labels"],
    ):

        x1, y1, x2, y2 = [
            float(v)
            for v in box
        ]

        x1 *= image.width
        x2 *= image.width
        y1 *= image.height
        y2 *= image.height

        label_id = int(
            label.item()
        )

        class_name = CLASS_NAMES[
            label_id
        ]

        text = (
            f"{class_name} "
            f"{float(score.item()):.2f}"
        )

        # Bounding box
        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline="red",
            width=3,
        )

        # Label background
        try:

            text_box = draw.textbbox(
                (x1, y1),
                text,
                font=font,
            )

            text_w = (
                text_box[2]
                -
                text_box[0]
            )

            text_h = (
                text_box[3]
                -
                text_box[1]
            )

        except Exception:

            text_w = len(text) * 8
            text_h = 18

        label_y1 = max(
            0,
            y1 - text_h - 6
        )

        label_y2 = (
            label_y1
            +
            text_h
            +
            6
        )

        draw.rectangle(
            [
                x1,
                label_y1,
                x1 + text_w + 8,
                label_y2,
            ],
            fill="red",
        )

        draw.text(
            (
                x1 + 4,
                label_y1 + 3
            ),
            text,
            fill="white",
            font=font,
        )

    return np.array(
        image
    )


# ============================================================
# HEATMAP OVERLAY
# ============================================================

def overlay_heatmap(
    image_rgb,
    heatmap,
):
    """
    Overlay the feature-activation heatmap.
    """

    if heatmap is None:
        return image_rgb

    heat_uint8 = (
        np.clip(
            heatmap,
            0,
            1
        )
        *
        255
    ).astype(
        np.uint8
    )

    colored = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    base = image_rgb.copy()

    output = cv2.addWeighted(
        base,
        0.55,
        colored,
        0.45,
        0
    )

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🌊 AquaYOLO"
    )

    st.caption(
        "AI-Based Underwater Organism Detection"
        " and Species Classification"
    )

    st.divider()

    st.subheader(
        "Model Settings"
    )

    weights_input = st.text_input(
        "Model checkpoint",
        value=str(
            DEFAULT_WEIGHTS
        ),
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=0.90,
        value=0.50,
        step=0.01,
    )

    nms_iou = st.slider(
        "NMS IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )

    st.divider()

    st.subheader(
        "System"
    )

    if torch.cuda.is_available():

        st.success(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )

    else:

        st.warning(
            "Running on CPU"
        )

    st.caption(
        f"Classes: {NUM_CLASSES}"
    )

    st.caption(
        "Input: 640 × 640"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌊 AquaYOLO"
)

st.subheader(
    "AI-Based Underwater Organism Detection "
    "and Species Intelligence"
)

st.write(
    "Upload an underwater image to enhance visibility, "
    "detect marine organisms, inspect confidence scores, "
    "and explore Aqua-Attn feature activation."
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

weights_path = Path(
    weights_input
)

if not weights_path.exists():

    st.error(
        "Model checkpoint not found:\n\n"
        f"{weights_path}"
    )

    st.info(
        "Train AquaYOLO first or select a valid "
        "best.pt checkpoint in the sidebar."
    )

    st.stop()

try:

    model, device = load_model(
        str(weights_path.resolve())
    )

except Exception as exc:

    st.error(
        "Could not load AquaYOLO checkpoint."
    )

    st.exception(
        exc
    )

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload underwater image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
)


# ============================================================
# MAIN APP
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an underwater image to begin."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "AquaYOLO"
        )

    with col2:

        st.metric(
            "Classes",
            "26"
        )

    with col3:

        st.metric(
            "Input",
            "640×640"
        )

else:

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    raw_bytes = uploaded_file.read()

    image = Image.open(
        io.BytesIO(raw_bytes)
    ).convert(
        "RGB"
    )

    image_rgb = np.array(
        image
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with st.spinner(
        "Enhancing underwater image and running AquaYOLO..."
    ):

        try:

            (
                enhanced,
                predictions,
                activation,
            ) = run_inference(
                model=model,
                device=device,
                image_rgb=image_rgb,
                confidence=confidence,
                nms_iou=nms_iou,
            )

        except Exception as exc:

            st.error(
                "Inference failed."
            )

            st.exception(
                exc
            )

            st.stop()

    # --------------------------------------------------------
    # Resize original for display
    # --------------------------------------------------------

    original_display = cv2.resize(
        image_rgb,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # Detection image
    # --------------------------------------------------------

    detection_image = draw_detections(
        enhanced,
        predictions,
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap = make_aqua_attention_heatmap(
        activation,
        output_size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
    )

    heatmap_overlay = overlay_heatmap(
        enhanced,
        heatmap,
    )

    # ========================================================
    # IMAGE RESULTS
    # ========================================================

    st.subheader(
        "Visual Analysis"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.markdown(
            "### Original"
        )

        st.image(
            original_display,
            use_container_width=True,
        )

    with col2:

        st.markdown(
            "### AquaYOLO Enhanced"
        )

        st.image(
            enhanced,
            use_container_width=True,
        )

    with col3:

        st.markdown(
            "### Detection"
        )

        st.image(
            detection_image,
            use_container_width=True,
        )

    # ========================================================
    # AQUA-ATTN VISUALIZATION
    # ========================================================

    st.divider()

    st.subheader(
        "Aqua-Attn Feature Visualization"
    )

    st.write(
        "The visualization below shows a PCA-based "
        "feature-activation heatmap derived from the "
        "Aqua-Attn output feature map."
    )

    st.image(
        heatmap_overlay,
        caption=(
            "Aqua-Attn activation visualization"
        ),
        use_container_width=True,
    )

    # ========================================================
    # DETECTION SUMMARY
    # ========================================================

    st.divider()

    detection_count = len(
        predictions["scores"]
    )

    st.subheader(
        "Detection Summary"
    )

    metric1, metric2 = st.columns(
        2
    )

    with metric1:

        st.metric(
            "Detections",
            detection_count
        )

    with metric2:

        if detection_count > 0:

            avg_conf = float(
                predictions["scores"]
                .mean()
                .item()
            )

        else:

            avg_conf = 0.0

        st.metric(
            "Average confidence",
            f"{avg_conf:.2%}"
        )

    # ========================================================
    # DETECTION TABLE
    # ========================================================

    if detection_count == 0:

        st.warning(
            "No detections passed the current "
            "confidence threshold."
        )

        st.caption(
            "For the current early baseline checkpoint, "
            "try lowering the confidence threshold for "
            "demonstration purposes."
        )

    else:

        st.subheader(
            "Detected Organisms"
        )

        for index, (
            box,
            score,
            label,
        ) in enumerate(
            zip(
                predictions["boxes"],
                predictions["scores"],
                predictions["labels"],
            ),
            start=1,
        ):

            class_id = int(
                label.item()
            )

            class_name = CLASS_NAMES[
                class_id
            ]

            score_value = float(
                score.item()
            )

            metadata = SPECIES_METADATA.get(
                class_name,
                {
                    "common_name": class_name,
                    "scientific_name": "Not available",
                    "taxonomy": "Not available",
                    "habitat": "Not available",
                    "toxicity": "Not available",
                    "edibility": "Not available",
                    "description": "No metadata available.",
                },
            )

            with st.expander(
                f"{index}. "
                f"{metadata['common_name']} "
                f"— {score_value:.1%} confidence",
                expanded=(
                    index == 1
                ),
            ):

                left, right = st.columns(
                    2
                )

                with left:

                    st.markdown(
                        f"**Scientific name:** "
                        f"{metadata['scientific_name']}"
                    )

                    st.markdown(
                        f"**Taxonomy:** "
                        f"{metadata['taxonomy']}"
                    )

                    st.markdown(
                        f"**Habitat:** "
                        f"{metadata['habitat']}"
                    )

                    st.markdown(
                        f"**Confidence:** "
                        f"{score_value:.2%}"
                    )

                with right:

                    st.markdown(
                        f"**Toxicity / safety:** "
                        f"{metadata['toxicity']}"
                    )

                    st.markdown(
                        f"**Edibility:** "
                        f"{metadata['edibility']}"
                    )

                    st.markdown(
                        f"**Description:** "
                        f"{metadata['description']}"
                    )

                # Normalized coordinates.
                x1, y1, x2, y2 = [
                    float(v)
                    for v in box
                ]

                st.caption(
                    "Bounding box "
                    f"(normalized): "
                    f"x1={x1:.3f}, "
                    f"y1={y1:.3f}, "
                    f"x2={x2:.3f}, "
                    f"y2={y2:.3f}"
                )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    with st.expander(
        "About AquaYOLO"
    ):

        st.markdown(
            """
            **AquaYOLO pipeline**

            1. Underwater image
            2. RGB → LAB conversion
            3. L-channel CLAHE
            4. Red-spectrum compensation
            5. CSPDarknet backbone
            6. SPPF
            7. Aqua-Attn
               - Spectral Attenuation Compensator (SAC)
               - Turbidity Contrast Gate (TCG)
            8. PANet multi-scale fusion
            9. Decoupled anchor-free detection head
            10. Species metadata lookup
            """
        )

    st.caption(
        "Current demo checkpoint: "
        f"{weights_path.name}"
    )