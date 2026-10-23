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
    / "AquaYOLO_26class_final"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "AquaYOLO_26class_balanced"
)


# ============================================================
# CLASSES
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


NUM_CLASSES = len(CLASS_NAMES)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# READ ONE IMAGE/LABEL PAIR
# ============================================================

def read_item(image_path, label_path):

    if not label_path.exists():
        return None

    classes = set()

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    # Empty labels are deliberately excluded.
    if not any(line.strip() for line in lines):
        return None

    for line in lines:

        parts = line.strip().split()

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

    if not classes:
        return None

    return {
        "image": image_path,
        "label": label_path,
        "classes": classes,
    }


# ============================================================
# COLLECT ALL VALID DATA
# ============================================================

def collect_items():

    items = []

    for split in [
        "train",
        "val",
        "test",
    ]:

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

        for image_path in sorted(
            image_dir.iterdir()
        ):

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


# ============================================================
# CLASS -> IMAGE INDEX
# ============================================================

def build_class_index(items):

    index = defaultdict(list)

    for idx, item in enumerate(items):

        for class_id in item["classes"]:
            index[class_id].append(idx)

    return index


# ============================================================
# CHOOSE COVERAGE IMAGE
# ============================================================

def choose_coverage_item(
    candidates,
    items,
    current_classes,
    class_frequency,
    rng,
):

    best_idx = None
    best_score = None

    for idx in candidates:

        classes = items[idx]["classes"]

        # Prioritize classes that are not yet represented.
        new_classes = (
            classes
            -
            current_classes
        )

        coverage_score = sum(
            1.0
            /
            max(
                class_frequency[c],
                1,
            )
            for c in new_classes
        )

        # Encourage multi-class images because one image
        # can improve coverage of several categories.
        multi_class_bonus = (
            0.25
            *
            len(new_classes)
        )

        score = (
            coverage_score
            +
            multi_class_bonus
            +
            rng.random()
            *
            1e-6
        )

        if (
            best_score is None
            or score > best_score
        ):

            best_score = score
            best_idx = idx

    return best_idx


# ============================================================
# FORCE EACH CLASS INTO VAL AND TEST
# ============================================================

def force_class_coverage(
    items,
    class_index,
    val_set,
    test_set,
    train_pool,
    rng,
):

    classes = sorted(
        range(NUM_CLASSES),
        key=lambda c: len(
            class_index[c]
        ),
    )

    # --------------------------------------------------------
    # Validation coverage
    # --------------------------------------------------------

    val_classes = set()

    for idx in val_set:
        val_classes.update(
            items[idx]["classes"]
        )

    for class_id in classes:

        if class_id in val_classes:
            continue

        candidates = [
            idx
            for idx in class_index[class_id]
            if idx in train_pool
        ]

        if not candidates:
            continue

        frequency = {
            c: len(class_index[c])
            for c in range(NUM_CLASSES)
        }

        chosen = choose_coverage_item(
            candidates,
            items,
            val_classes,
            frequency,
            rng,
        )

        val_set.add(chosen)
        train_pool.remove(chosen)
        val_classes.update(
            items[chosen]["classes"]
        )

    # --------------------------------------------------------
    # Test coverage
    # --------------------------------------------------------

    test_classes = set()

    for idx in test_set:
        test_classes.update(
            items[idx]["classes"]
        )

    for class_id in classes:

        if class_id in test_classes:
            continue

        candidates = [
            idx
            for idx in class_index[class_id]
            if idx in train_pool
        ]

        if not candidates:
            continue

        frequency = {
            c: len(class_index[c])
            for c in range(NUM_CLASSES)
        }

        chosen = choose_coverage_item(
            candidates,
            items,
            test_classes,
            frequency,
            rng,
        )

        test_set.add(chosen)
        train_pool.remove(chosen)
        test_classes.update(
            items[chosen]["classes"]
        )


# ============================================================
# FILL SPLITS TO TARGET SIZE
# ============================================================

def fill_split(
    target_size,
    split_set,
    train_pool,
    items,
    class_index,
    rng,
):

    if len(split_set) >= target_size:
        return

    current_counts = defaultdict(int)

    for idx in split_set:

        for class_id in items[idx]["classes"]:
            current_counts[class_id] += 1

    # Desired image-level class count in this split.
    desired = {}

    fraction = (
        target_size
        /
        max(
            len(items),
            1,
        )
    )

    for class_id in range(NUM_CLASSES):

        desired[class_id] = max(
            1,
            round(
                len(class_index[class_id])
                *
                fraction
            ),
        )

    while (
        len(split_set)
        <
        target_size
        and
        train_pool
    ):

        best_idx = None
        best_score = None

        sample_indices = list(
            train_pool
        )

        # Evaluate all candidates.
        for idx in sample_indices:

            score = 0.0

            for class_id in items[idx]["classes"]:

                deficit = (
                    desired[class_id]
                    -
                    current_counts[class_id]
                )

                frequency = max(
                    len(
                        class_index[class_id]
                    ),
                    1,
                )

                score += (
                    max(
                        deficit,
                        0,
                    )
                    /
                    frequency
                )

            # Small preference for multi-class images.
            score += (
                0.01
                *
                len(
                    items[idx]["classes"]
                )
            )

            score += rng.random() * 1e-6

            if (
                best_score is None
                or score > best_score
            ):

                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        split_set.add(
            best_idx
        )

        train_pool.remove(
            best_idx
        )

        for class_id in items[
            best_idx
        ]["classes"]:

            current_counts[
                class_id
            ] += 1


# ============================================================
# BUILD ONE CANDIDATE SPLIT
# ============================================================

def build_candidate(
    items,
    seed,
):

    rng = random.Random(
        seed
    )

    total = len(items)

    target_val = round(
        total * 0.15
    )

    target_test = round(
        total * 0.15
    )

    indices = list(
        range(total)
    )

    rng.shuffle(
        indices
    )

    train_pool = set(
        indices
    )

    val_set = set()
    test_set = set()

    class_index = (
        build_class_index(items)
    )

    # --------------------------------------------------------
    # Must have at least one image per class in val/test.
    # --------------------------------------------------------

    force_class_coverage(
        items,
        class_index,
        val_set,
        test_set,
        train_pool,
        rng,
    )

    # --------------------------------------------------------
    # Fill validation and test.
    # --------------------------------------------------------

    fill_split(
        target_val,
        val_set,
        train_pool,
        items,
        class_index,
        rng,
    )

    fill_split(
        target_test,
        test_set,
        train_pool,
        items,
        class_index,
        rng,
    )

    train_set = set(
        train_pool
    )

    return (
        train_set,
        val_set,
        test_set,
    )


# ============================================================
# SCORE SPLIT
# ============================================================

def split_score(
    items,
    train_set,
    val_set,
    test_set,
):

    # Missing-class penalty.
    val_classes = set()
    test_classes = set()

    for idx in val_set:
        val_classes.update(
            items[idx]["classes"]
        )

    for idx in test_set:
        test_classes.update(
            items[idx]["classes"]
        )

    missing_val = (
        NUM_CLASSES
        -
        len(val_classes)
    )

    missing_test = (
        NUM_CLASSES
        -
        len(test_classes)
    )

    coverage_penalty = (
        missing_val * 100000
        +
        missing_test * 100000
    )

    # Size penalty.
    total = len(items)

    expected_val = total * 0.15
    expected_test = total * 0.15

    size_penalty = (
        abs(
            len(val_set)
            -
            expected_val
        )
        +
        abs(
            len(test_set)
            -
            expected_test
        )
    )

    # Prefer balanced class proportions.
    class_index = build_class_index(
        items
    )

    distribution_penalty = 0.0

    for class_id in range(NUM_CLASSES):

        total_class = len(
            class_index[class_id]
        )

        if total_class == 0:
            return float(
                "inf"
            )

        expected_val_class = (
            total_class * 0.15
        )

        expected_test_class = (
            total_class * 0.15
        )

        actual_val_class = sum(
            1
            for idx in val_set
            if class_id
            in items[idx]["classes"]
        )

        actual_test_class = sum(
            1
            for idx in test_set
            if class_id
            in items[idx]["classes"]
        )

        distribution_penalty += abs(
            actual_val_class
            -
            expected_val_class
        )

        distribution_penalty += abs(
            actual_test_class
            -
            expected_test_class
        )

    return (
        coverage_penalty
        +
        size_penalty
        +
        distribution_penalty
    )


# ============================================================
# WRITE DATASET
# ============================================================

def write_split(
    split_name,
    indices,
    items,
):

    image_dir = (
        OUTPUT
        / "images"
        / split_name
    )

    label_dir = (
        OUTPUT
        / "labels"
        / split_name
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for idx in sorted(indices):

        item = items[idx]

        destination_image = (
            image_dir
            /
            item["image"].name
        )

        destination_label = (
            label_dir
            /
            item["label"].name
        )

        shutil.copy2(
            item["image"],
            destination_image,
        )

        shutil.copy2(
            item["label"],
            destination_label,
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
    print("AQUAYOLO FINAL CLASS-AWARE DATASET REBALANCER")
    print("=" * 80)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE}"
        )

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists:\n{OUTPUT}\n\n"
            "Delete it only if you intentionally want "
            "to rebuild it."
        )

    print()
    print("Source:")
    print(SOURCE)

    print()
    print("Output:")
    print(OUTPUT)

    # --------------------------------------------------------
    # Collect all non-empty valid images.
    # --------------------------------------------------------

    items = collect_items()

    print()
    print(
        "Valid annotated images:",
        len(items),
    )

    if len(items) < 100:
        raise RuntimeError(
            "Too few valid images."
        )

    class_index = build_class_index(
        items
    )

    print()
    print("Available classes:")

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{class_id:2d} "
            f"{CLASS_NAMES[class_id]:25s} "
            f"{len(class_index[class_id])}"
        )

    # --------------------------------------------------------
    # Try many deterministic seeds.
    # --------------------------------------------------------

    best = None
    best_score = float("inf")

    print()
    print(
        "Searching for a balanced split..."
    )

    for seed in range(200):

        train_set, val_set, test_set = (
            build_candidate(
                items,
                seed,
            )
        )

        score = split_score(
            items,
            train_set,
            val_set,
            test_set,
        )

        if score < best_score:

            best_score = score

            best = (
                seed,
                train_set,
                val_set,
                test_set,
            )

        if score == 0:
            break

    if best is None:
        raise RuntimeError(
            "Could not create a valid split."
        )

    seed, train_set, val_set, test_set = (
        best
    )

    # --------------------------------------------------------
    # Final coverage verification.
    # --------------------------------------------------------

    train_classes = set()
    val_classes = set()
    test_classes = set()

    for idx in train_set:
        train_classes.update(
            items[idx]["classes"]
        )

    for idx in val_set:
        val_classes.update(
            items[idx]["classes"]
        )

    for idx in test_set:
        test_classes.update(
            items[idx]["classes"]
        )

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

    if missing_train:
        raise RuntimeError(
            f"Missing train classes: {missing_train}"
        )

    if missing_val:
        raise RuntimeError(
            f"Missing validation classes: {missing_val}"
        )

    if missing_test:
        raise RuntimeError(
            f"Missing test classes: {missing_test}"
        )

    # --------------------------------------------------------
    # Create output.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SELECTED SPLIT")
    print("=" * 80)

    print(
        "Seed:",
        seed,
    )

    print(
        "Train:",
        len(train_set),
    )

    print(
        "Val:",
        len(val_set),
    )

    print(
        "Test:",
        len(test_set),
    )

    write_split(
        "train",
        train_set,
        items,
    )

    write_split(
        "val",
        val_set,
        items,
    )

    write_split(
        "test",
        test_set,
        items,
    )

    write_yaml()

    print()
    print("=" * 80)
    print("FINAL BALANCED DATASET CREATED")
    print("=" * 80)

    print(OUTPUT)

    print()
    print(
        "All 26 classes are present in "
        "train, validation and test."
    )

    print()
    print(
        "Next step: run audit_dataset.py "
        "against this new output."
    )


if __name__ == "__main__":
    main()