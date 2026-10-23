from pathlib import Path
from collections import defaultdict
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE = (
    PROJECT_ROOT
    / "data"
    / "AquaYOLO_26class_final"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "AquaYOLO_26class_balanced"
)

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

NUM_CLASSES = 26

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def read_item(image_path, label_path):
    if not label_path.exists():
        return None

    lines = [
        x.strip()
        for x in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if not lines:
        return None

    classes = set()

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            return None

        try:
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            return None

        if not (
            0 <= class_id < NUM_CLASSES
            and 0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            return None

        classes.add(class_id)

    return {
        "image": image_path,
        "label": label_path,
        "classes": classes,
    }


def collect_all():
    items = []

    for split in ["train", "val", "test"]:

        image_dir = SOURCE / "images" / split
        label_dir = SOURCE / "labels" / split

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            item = read_item(
                image_path,
                label_path,
            )

            if item is not None:
                items.append(item)

    return items


def write_split(name, items):

    image_dir = (
        OUTPUT
        / "images"
        / name
    )

    label_dir = (
        OUTPUT
        / "labels"
        / name
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in items:

        shutil.copy2(
            item["image"],
            image_dir / item["image"].name,
        )

        shutil.copy2(
            item["label"],
            label_dir / item["label"].name,
        )


def write_yaml():

    text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
    )

    for i, name in enumerate(CLASS_NAMES):
        text += f"  {i}: {name}\n"

    (
        OUTPUT / "data.yaml"
    ).write_text(
        text,
        encoding="utf-8",
    )


def main():

    print("=" * 80)
    print("FAST AQUAYOLO 26-CLASS REBALANCER")
    print("=" * 80)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE}"
        )

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists:\n{OUTPUT}"
        )

    items = collect_all()

    print()
    print(
        "Valid annotated images:",
        len(items),
    )

    rng = random.Random(42)

    # --------------------------------------------------------
    # Build class -> candidate image index.
    # --------------------------------------------------------

    class_to_items = defaultdict(list)

    for idx, item in enumerate(items):
        for class_id in item["classes"]:
            class_to_items[class_id].append(idx)

    # --------------------------------------------------------
    # Remove duplicate index selection.
    # --------------------------------------------------------

    remaining = set(range(len(items)))

    val_indices = set()
    test_indices = set()

    # --------------------------------------------------------
    # First guarantee every class exists in VAL and TEST.
    #
    # Rare classes are handled first.
    # --------------------------------------------------------

    classes_by_rarity = sorted(
        range(NUM_CLASSES),
        key=lambda c: len(
            class_to_items[c]
        ),
    )

    # Validation coverage.
    val_classes = set()

    for class_id in classes_by_rarity:

        if class_id in val_classes:
            continue

        candidates = [
            idx
            for idx in class_to_items[class_id]
            if idx in remaining
        ]

        if not candidates:
            continue

        # Prefer an image that contains several
        # currently missing classes.
        candidates.sort(
            key=lambda idx: len(
                items[idx]["classes"]
                -
                val_classes
            ),
            reverse=True,
        )

        chosen = candidates[0]

        val_indices.add(chosen)
        remaining.remove(chosen)

        val_classes.update(
            items[chosen]["classes"]
        )

    # Test coverage.
    test_classes = set()

    for class_id in classes_by_rarity:

        if class_id in test_classes:
            continue

        candidates = [
            idx
            for idx in class_to_items[class_id]
            if idx in remaining
        ]

        if not candidates:
            continue

        candidates.sort(
            key=lambda idx: len(
                items[idx]["classes"]
                -
                test_classes
            ),
            reverse=True,
        )

        chosen = candidates[0]

        test_indices.add(chosen)
        remaining.remove(chosen)

        test_classes.update(
            items[chosen]["classes"]
        )

    # --------------------------------------------------------
    # Target sizes.
    # --------------------------------------------------------

    total = len(items)

    target_val = round(
        total * 0.15
    )

    target_test = round(
        total * 0.15
    )

    # --------------------------------------------------------
    # Fill validation.
    # --------------------------------------------------------

    remaining_list = list(remaining)
    rng.shuffle(remaining_list)

    while (
        len(val_indices) < target_val
        and remaining_list
    ):

        idx = remaining_list.pop()

        val_indices.add(idx)
        remaining.remove(idx)

    # --------------------------------------------------------
    # Fill test.
    # --------------------------------------------------------

    remaining_list = list(remaining)
    rng.shuffle(remaining_list)

    while (
        len(test_indices) < target_test
        and remaining_list
    ):

        idx = remaining_list.pop()

        test_indices.add(idx)
        remaining.remove(idx)

    # --------------------------------------------------------
    # Everything remaining -> train.
    # --------------------------------------------------------

    train_indices = remaining

    train_items = [
        items[i]
        for i in sorted(train_indices)
    ]

    val_items = [
        items[i]
        for i in sorted(val_indices)
    ]

    test_items = [
        items[i]
        for i in sorted(test_indices)
    ]

    # --------------------------------------------------------
    # Verify class coverage.
    # --------------------------------------------------------

    def classes_in(split_items):
        found = set()

        for item in split_items:
            found.update(
                item["classes"]
            )

        return found

    train_classes = classes_in(train_items)
    val_classes = classes_in(val_items)
    test_classes = classes_in(test_items)

    missing_train = (
        set(range(NUM_CLASSES))
        -
        train_classes
    )

    missing_val = (
        set(range(NUM_CLASSES))
        -
        val_classes
    )

    missing_test = (
        set(range(NUM_CLASSES))
        -
        test_classes
    )

    print()
    print("=" * 80)
    print("SPLIT")
    print("=" * 80)

    print(
        "Train:",
        len(train_items)
    )

    print(
        "Val:",
        len(val_items)
    )

    print(
        "Test:",
        len(test_items)
    )

    print()
    print(
        "Missing train classes:",
        sorted(missing_train),
    )

    print(
        "Missing val classes:",
        sorted(missing_val),
    )

    print(
        "Missing test classes:",
        sorted(missing_test),
    )

    if (
        missing_train
        or missing_val
        or missing_test
    ):
        raise RuntimeError(
            "Could not achieve complete 26-class coverage."
        )

    # --------------------------------------------------------
    # Write dataset.
    # --------------------------------------------------------

    print()
    print("Writing train...")
    write_split(
        "train",
        train_items,
    )

    print("Writing val...")
    write_split(
        "val",
        val_items,
    )

    print("Writing test...")
    write_split(
        "test",
        test_items,
    )

    write_yaml()

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "Output:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()