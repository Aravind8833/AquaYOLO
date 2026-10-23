from pathlib import Path
from collections import defaultdict
import random
import shutil


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE = (
    PROJECT_ROOT
    / "data"
    / "AquaYOLO_26class_merged"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "AquaYOLO_26class_final"
)


# ============================================================
# CLASS NAMES
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


FJORDFISH_CLASSES = set(range(14, 25))


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

def read_classes(label_path):
    classes = set()

    if not label_path.exists():
        return classes

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        classes.add(class_id)

    return classes


def collect_fjord_images(split):
    """
    Collect FjordFish images from the temporary merged dataset.
    """

    image_dir = (
        SOURCE
        / "images"
        / split
    )

    label_dir = (
        SOURCE
        / "labels"
        / split
    )

    items = []

    for image_path in sorted(image_dir.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if not image_path.stem.startswith("fjord_"):
            continue

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        classes = read_classes(
            label_path
        )

        target_classes = (
            classes
            & FJORDFISH_CLASSES
        )

        if not target_classes:
            continue

        items.append(
            {
                "image": image_path,
                "label": label_path,
                "classes": target_classes,
            }
        )

    return items


def split_fjordfish(items):
    """
    Deterministic 70/15/15 split.

    Repeats with different seeds until every selected
    FjordFish class appears in both validation and test.
    """

    required = FJORDFISH_CLASSES

    total = len(items)

    train_count = int(
        total * 0.70
    )

    val_count = int(
        total * 0.15
    )

    test_count = (
        total
        -
        train_count
        -
        val_count
    )

    for seed in range(1000):

        rng = random.Random(seed)

        shuffled = items.copy()

        rng.shuffle(
            shuffled
        )

        train_items = shuffled[
            :train_count
        ]

        val_items = shuffled[
            train_count:
            train_count + val_count
        ]

        test_items = shuffled[
            train_count + val_count:
        ]

        val_classes = set()

        for item in val_items:
            val_classes.update(
                item["classes"]
            )

        test_classes = set()

        for item in test_items:
            test_classes.update(
                item["classes"]
            )

        if (
            required <= val_classes
            and
            required <= test_classes
        ):

            print(
                f"FjordFish split seed: {seed}"
            )

            return {
                "train": train_items,
                "val": val_items,
                "test": test_items,
            }

    raise RuntimeError(
        "Could not create a split where all "
        "11 FjordFish classes appear in both "
        "validation and test."
    )


def copy_item(
    item,
    split,
):

    destination_image_dir = (
        OUTPUT
        / "images"
        / split
    )

    destination_label_dir = (
        OUTPUT
        / "labels"
        / split
    )

    destination_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        item["image"],
        destination_image_dir
        /
        item["image"].name,
    )

    shutil.copy2(
        item["label"],
        destination_label_dir
        /
        item["label"].name,
    )


def copy_baseline():

    print()
    print("=" * 80)
    print("COPYING BASELINE DATA")
    print("=" * 80)

    for split in [
        "train",
        "val",
        "test",
    ]:

        source_image_dir = (
            SOURCE
            / "images"
            / split
        )

        source_label_dir = (
            SOURCE
            / "labels"
            / split
        )

        destination_image_dir = (
            OUTPUT
            / "images"
            / split
        )

        destination_label_dir = (
            OUTPUT
            / "labels"
            / split
        )

        destination_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        count = 0

        for image_path in sorted(
            source_image_dir.iterdir()
        ):

            if not image_path.is_file():
                continue

            if not image_path.name.startswith(
                "base_"
            ):
                continue

            shutil.copy2(
                image_path,
                destination_image_dir
                /
                image_path.name,
            )

            label_path = (
                source_label_dir
                /
                f"{image_path.stem}.txt"
            )

            if label_path.exists():

                shutil.copy2(
                    label_path,
                    destination_label_dir
                    /
                    label_path.name,
                )

            else:

                (
                    destination_label_dir
                    /
                    f"{image_path.stem}.txt"
                ).write_text(
                    "",
                    encoding="utf-8",
                )

            count += 1

        print(
            f"Baseline {split}: "
            f"{count} images"
        )


def write_yaml():

    text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
    )

    for class_id, name in enumerate(
        CLASS_NAMES
    ):

        text += (
            f"  {class_id}: {name}\n"
        )

    (
        OUTPUT / "data.yaml"
    ).write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AQUAYOLO FINAL 26-CLASS DATASET BUILDER")
    print("=" * 80)

    print()
    print("Source:")
    print(SOURCE)

    print()
    print("Output:")
    print(OUTPUT)

    if not SOURCE.exists():

        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE}"
        )

    if OUTPUT.exists():

        raise FileExistsError(
            f"Output already exists:\n{OUTPUT}\n\n"
            "Remove it only if you intentionally "
            "want to rebuild the final dataset."
        )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    copy_baseline()

    # --------------------------------------------------------
    # Collect FjordFish
    #
    # The temporary merge placed FjordFish images in
    # the train/val folders. We now redistribute them.
    # --------------------------------------------------------

    fjord_items = []

    fjord_items.extend(
        collect_fjord_images(
            "train"
        )
    )

    fjord_items.extend(
        collect_fjord_images(
            "val"
        )
    )

    print()
    print(
        "FjordFish images available for split:",
        len(fjord_items),
    )

    if not fjord_items:

        raise RuntimeError(
            "No FjordFish images found."
        )

    # --------------------------------------------------------
    # Split 70 / 15 / 15
    # --------------------------------------------------------

    split_items = split_fjordfish(
        fjord_items
    )

    print()
    print("=" * 80)
    print("FJORDFISH SPLIT")
    print("=" * 80)

    for split in [
        "train",
        "val",
        "test",
    ]:

        items = split_items[
            split
        ]

        for item in items:

            copy_item(
                item,
                split,
            )

        print(
            f"{split:5s}: "
            f"{len(items)} images"
        )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    write_yaml()

    print()
    print("=" * 80)
    print("FINAL DATASET CREATED")
    print("=" * 80)

    print(
        OUTPUT
    )

    print()
    print(
        "Run audit_dataset.py against this dataset "
        "before training."
    )


if __name__ == "__main__":
    main()