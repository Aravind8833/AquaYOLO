from pathlib import Path
import shutil


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE = (
    PROJECT_ROOT
    / "data"
    / "AquaYOLO_baseline_species26"
)

FJORDFISH = (
    Path.home()
    / "Downloads"
    / "FjordFish"
    / "FjordFish"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "AquaYOLO_26class_merged"
)


# ============================================================
# AQUAYOLO CLASSES
# ============================================================

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


# ============================================================
# FJORDFISH SOURCE -> AQUAYOLO TARGET
# ============================================================

FJORDFISH_MAP = {
    0: 14,   # ballan -> ballan_wrasse

    1: 15,   # cuckoo_female
    2: 15,   # cuckoo_male

    3: 16,   # corkwing_female
    4: 16,   # corkwing_male

    5: 17,   # goldsinny
    6: 18,   # rock_cook

    8: 19,   # cod
    9: 19,   # cod_yoy

    11: 20,  # pollack
    14: 21,  # whiting
    16: 22,  # spiny_dogfish
    18: 23,  # brown_crab
    24: 24,  # two_spot_goby
}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():

    for split in ["train", "val", "test"]:

        (
            OUTPUT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def copy_existing_split(split):

    source_images = (
        BASELINE
        / "images"
        / split
    )

    source_labels = (
        BASELINE
        / "labels"
        / split
    )

    dest_images = (
        OUTPUT
        / "images"
        / split
    )

    dest_labels = (
        OUTPUT
        / "labels"
        / split
    )

    images = [
        p
        for p in source_images.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    copied = 0

    for image_path in images:

        label_path = (
            source_labels
            / f"{image_path.stem}.txt"
        )

        new_image_name = (
            f"base_{image_path.name}"
        )

        new_label_name = (
            f"base_{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            dest_images / new_image_name,
        )

        if label_path.exists():

            shutil.copy2(
                label_path,
                dest_labels / new_label_name,
            )

        else:

            (
                dest_labels / new_label_name
            ).write_text(
                "",
                encoding="utf-8",
            )

        copied += 1

    print(
        f"Copied baseline {split}: "
        f"{copied} images"
    )


def merge_fjordfish_split(
    split,
):

    source_images = (
        FJORDFISH
        / "images"
        / split
    )

    source_labels = (
        FJORDFISH
        / "labels"
        / split
    )

    dest_images = (
        OUTPUT
        / "images"
        / split
    )

    dest_labels = (
        OUTPUT
        / "labels"
        / split
    )

    if not source_images.exists():

        print(
            f"FjordFish {split}: "
            f"image folder not found"
        )

        return

    images = [
        p
        for p in source_images.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    copied_images = 0
    kept_boxes = 0
    dropped_boxes = 0
    skipped_images = 0

    for image_path in images:

        label_path = (
            source_labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            skipped_images += 1
            continue

        output_lines = []

        raw_lines = (
            label_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

        for line in raw_lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                dropped_boxes += 1
                continue

            try:

                source_class = int(parts[0])

                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                dropped_boxes += 1
                continue

            # ------------------------------------------------
            # Keep only selected FjordFish species.
            # ------------------------------------------------

            if source_class not in FJORDFISH_MAP:

                dropped_boxes += 1
                continue

            target_class = FJORDFISH_MAP[
                source_class
            ]

            # ------------------------------------------------
            # Safety validation.
            # ------------------------------------------------

            if not (
                0.0 <= cx <= 1.0
                and
                0.0 <= cy <= 1.0
                and
                0.0 < w <= 1.0
                and
                0.0 < h <= 1.0
            ):

                dropped_boxes += 1
                continue

            output_lines.append(
                f"{target_class} "
                f"{cx:.6f} "
                f"{cy:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

            kept_boxes += 1

        # ----------------------------------------------------
        # Do not import images that have no selected target
        # species after filtering.
        # ----------------------------------------------------

        if not output_lines:

            continue

        new_stem = (
            f"fjord_{split}_{image_path.stem}"
        )

        new_image = (
            dest_images
            / f"{new_stem}{image_path.suffix.lower()}"
        )

        new_label = (
            dest_labels
            / f"{new_stem}.txt"
        )

        shutil.copy2(
            image_path,
            new_image,
        )

        new_label.write_text(
            "\n".join(output_lines)
            + "\n",
            encoding="utf-8",
        )

        copied_images += 1

    print()
    print(
        f"FjordFish {split}:"
    )

    print(
        f"  images added : "
        f"{copied_images}"
    )

    print(
        f"  boxes kept   : "
        f"{kept_boxes}"
    )

    print(
        f"  boxes dropped: "
        f"{dropped_boxes}"
    )

    print(
        f"  skipped      : "
        f"{skipped_images}"
    )


def write_yaml():

    yaml_text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
    )

    for class_id, name in enumerate(
        CLASS_NAMES
    ):

        yaml_text += (
            f"  {class_id}: {name}\n"
        )

    (
        OUTPUT / "data.yaml"
    ).write_text(
        yaml_text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AQUAYOLO + FJORDFISH DATASET MERGE")
    print("=" * 80)

    print()
    print(
        "Baseline:",
        BASELINE,
    )

    print(
        "FjordFish:",
        FJORDFISH,
    )

    print(
        "Output:",
        OUTPUT,
    )

    if not BASELINE.exists():

        raise FileNotFoundError(
            f"Baseline dataset not found:\n{BASELINE}"
        )

    if not FJORDFISH.exists():

        raise FileNotFoundError(
            f"FjordFish dataset not found:\n{FJORDFISH}"
        )

    if OUTPUT.exists():

        raise FileExistsError(
            "Output dataset already exists.\n"
            f"{OUTPUT}\n\n"
            "Delete/rename it only if you intentionally "
            "want to rebuild it."
        )

    ensure_dirs()

    print()
    print("=" * 80)
    print("COPYING EXISTING AQUAYOLO DATASET")
    print("=" * 80)

    for split in [
        "train",
        "val",
        "test",
    ]:

        copy_existing_split(
            split
        )

    print()
    print("=" * 80)
    print("MERGING FJORDFISH")
    print("=" * 80)

    merge_fjordfish_split(
        "train"
    )

    merge_fjordfish_split(
        "val"
    )

    write_yaml()

    print()
    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)

    print()
    print(
        "New dataset:",
        OUTPUT,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Run the dataset audit before training."
    )


if __name__ == "__main__":
    main()