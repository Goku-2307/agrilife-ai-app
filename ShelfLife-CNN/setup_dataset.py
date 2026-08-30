from pathlib import Path
import shutil
import random

# ============================================================
# SETTINGS
# ============================================================

# Original dataset
SOURCE_DIR = Path(
    "raw_dataset/Fruit Freshness Dataset"
)

# Processed dataset
DEST_DIR = Path("dataset")

# Reproducible random split
RANDOM_SEED = 42

# README-required split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# CLASSES
# ============================================================

CLASS_MAPPING = {
    ("Apple", "Fresh"): "fresh_apple",
    ("Apple", "Rotten"): "rotten_apple",
    ("Banana", "Fresh"): "fresh_banana",
    ("Banana", "Rotten"): "rotten_banana"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 0.001

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)

# ============================================================
# CHECK SOURCE DATASET
# ============================================================

print("=" * 60)
print("       SHELFLIFE CNN - DATASET PREPARATION")
print("=" * 60)

print("\nSource:")
print(SOURCE_DIR.resolve())

print("\nDestination:")
print(DEST_DIR.resolve())

if not SOURCE_DIR.exists():

    print("\nERROR: Source dataset not found!")
    print("\nExpected structure:")
    print(
        "raw_dataset/Fruit Freshness Dataset/"
    )
    print("    Apple/")
    print("    Banana/")
    exit()

# ============================================================
# CLEAR EXISTING PROCESSED DATASET
# ============================================================

print("\nChecking existing dataset...")

if DEST_DIR.exists():

    print(
        "Existing dataset folder found."
    )

    print(
        "Removing old train/val/test contents..."
    )

    for split in ["train", "val", "test"]:

        split_dir = DEST_DIR / split

        if split_dir.exists():

            shutil.rmtree(split_dir)

# ============================================================
# CREATE FOLDERS
# ============================================================

for split in ["train", "val", "test"]:

    for class_name in CLASS_MAPPING.values():

        folder = (
            DEST_DIR
            / split
            / class_name
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

# ============================================================
# PROCESS EACH CLASS
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

total_train = 0
total_val = 0
total_test = 0

for (fruit, condition), class_name in CLASS_MAPPING.items():

    source_folder = (
        SOURCE_DIR
        / fruit
        / condition
    )

    print(
        f"\n{fruit} - {condition}"
    )

    if not source_folder.exists():

        print(
            f"ERROR: Folder not found: {source_folder}"
        )

        continue

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    images = [
        file
        for file in source_folder.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    # Sort first so the random seed gives reproducible results
    images.sort()

    # Shuffle
    random.shuffle(images)

    total = len(images)

    # --------------------------------------------------------
    # CALCULATE SPLIT
    # --------------------------------------------------------

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    train_images = images[
        :train_count
    ]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    # --------------------------------------------------------
    # PRINT COUNTS
    # --------------------------------------------------------

    print(f"Total : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")

    # --------------------------------------------------------
    # COPY FUNCTION
    # --------------------------------------------------------

    def copy_images(
        image_list,
        split_name
    ):

        destination = (
            DEST_DIR
            / split_name
            / class_name
        )

        for index, image_path in enumerate(
            image_list
        ):

            # Keep original extension
            new_name = (
                f"{class_name}_{index:04d}"
                f"{image_path.suffix.lower()}"
            )

            destination_file = (
                destination
                / new_name
            )

            shutil.copy2(
                image_path,
                destination_file
            )

    # --------------------------------------------------------
    # COPY IMAGES
    # --------------------------------------------------------

    copy_images(
        train_images,
        "train"
    )

    copy_images(
        val_images,
        "val"
    )

    copy_images(
        test_images,
        "test"
    )

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print(
    f"\nTraining images   : {total_train}"
)

print(
    f"Validation images : {total_val}"
)

print(
    f"Testing images    : {total_test}"
)

print(
    f"Total images      : "
    f"{total_train + total_val + total_test}"
)

# ============================================================
# FINAL FOLDER CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET STRUCTURE")
print("=" * 60)

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    for class_name in CLASS_MAPPING.values():

        folder = (
            DEST_DIR
            / split
            / class_name
        )

        count = sum(
            1
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        )

        print(
            f"  {class_name:<15} : {count}"
        )

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("READY FOR CNN TRAINING")
print("=" * 60)

print("\nClasses:")
print("  1. fresh_apple")
print("  2. rotten_apple")
print("  3. fresh_banana")
print("  4. rotten_banana")

print("\nNext step:")
print("Run:")
print("    python train.py")