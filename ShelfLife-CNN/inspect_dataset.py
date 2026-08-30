from pathlib import Path
from PIL import Image

# ============================================================
# LOCAL DATASET LOCATION
# ============================================================

DATASET_DIR = Path("raw_dataset/Fruit Freshness Dataset")

# We only use Apple and Banana
classes = {
    "Apple": ["Fresh", "Rotten"],
    "Banana": ["Fresh", "Rotten"]
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# ============================================================
# COUNTERS
# ============================================================

total_images = 0
valid_images = 0
corrupted_images = 0

# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("       SHELFLIFE CNN - DATASET INSPECTION")
print("=" * 60)

print("\nDataset path:")
print(DATASET_DIR.resolve())

# ============================================================
# CHECK DATASET LOCATION
# ============================================================

if not DATASET_DIR.exists():
    print("\nERROR: Dataset folder not found!")
    print("\nExpected:")
    print("raw_dataset/Fruit Freshness Dataset")
    exit()

# ============================================================
# INSPECT APPLE + BANANA
# ============================================================

for fruit, conditions in classes.items():

    print("\n" + "=" * 60)
    print(f"{fruit.upper()}")
    print("=" * 60)

    for condition in conditions:

        folder = DATASET_DIR / fruit / condition

        print(f"\n{fruit} - {condition}")

        if not folder.exists():
            print("  ERROR: Folder not found!")
            continue

        images = [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"  Images found: {len(images)}")

        total_images += len(images)

        class_valid = 0
        class_corrupted = 0

        for image_path in images:

            try:

                with Image.open(image_path) as img:
                    img.verify()

                class_valid += 1
                valid_images += 1

            except Exception:

                class_corrupted += 1
                corrupted_images += 1

                print(
                    f"  CORRUPTED: {image_path.name}"
                )

        print(f"  Valid images: {class_valid}")
        print(f"  Corrupted images: {class_corrupted}")

# ============================================================
# CHECK STRAWBERRY
# ============================================================

strawberry_dir = DATASET_DIR / "Strawberry"

print("\n" + "=" * 60)
print("STRAWBERRY")
print("=" * 60)

if strawberry_dir.exists():

    strawberry_total = 0

    for condition in ["Fresh", "Rotten"]:

        folder = strawberry_dir / condition

        if folder.exists():

            count = sum(
                1
                for file in folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"Strawberry - {condition}: {count}")

            strawberry_total += count

    print(f"\nTotal Strawberry images: {strawberry_total}")
    print("Strawberry will NOT be used in this CNN.")

else:

    print("No Strawberry folder found.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nApple + Banana images : {total_images}")
print(f"Valid images          : {valid_images}")
print(f"Corrupted images      : {corrupted_images}")

print("\nClasses used by CNN:")

print("  1. fresh_apple")
print("  2. rotten_apple")
print("  3. fresh_banana")
print("  4. rotten_banana")

# ============================================================
# STATUS
# ============================================================

print("\n" + "=" * 60)

if corrupted_images == 0:
    print("STATUS: DATASET LOOKS GOOD")
else:
    print("STATUS: CORRUPTED IMAGES FOUND")

print("=" * 60)