import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# SHELFLIFE INTELLIGENCE - MODEL EVALUATION
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = "best_fruit_quality_model.pth"

CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = 224

BATCH_SIZE = 16


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("       SHELFLIFE INTELLIGENCE - MODEL EVALUATION")
    print("=" * 60)


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )


    # ========================================================
    # CHECK FILES
    # ========================================================

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DIR}"
        )


    if not Path(MODEL_PATH).exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )


    if not Path(CLASS_NAMES_PATH).exists():

        raise FileNotFoundError(
            f"Class names file not found: "
            f"{CLASS_NAMES_PATH}"
        )


    # ========================================================
    # LOAD CLASS NAMES
    # ========================================================

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as file:

        class_names = json.load(file)


    num_classes = len(class_names)


    print("\nClasses:")

    for index, name in enumerate(class_names):

        print(
            f"  {index}: {name}"
        )


    # ========================================================
    # TEST TRANSFORM
    # ========================================================

    test_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


    # ========================================================
    # LOAD TEST DATASET
    # ========================================================

    print("\nLoading test dataset...")


    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )


    # ========================================================
    # VERIFY CLASSES
    # ========================================================

    if test_dataset.classes != class_names:

        raise ValueError(
            "\nTest dataset classes do not "
            "match class_names.json.\n"
            f"Dataset classes: {test_dataset.classes}\n"
            f"Saved classes: {class_names}"
        )


    print(
        f"Test images: {len(test_dataset)}"
    )


    # ========================================================
    # TEST CLASS DISTRIBUTION
    # ========================================================

    print("\nTest class distribution:")


    for class_index, class_name in enumerate(
        class_names
    ):

        count = sum(
            1
            for _, label in test_dataset.samples
            if label == class_index
        )

        print(
            f"  {class_name:<15}: {count}"
        )


    # ========================================================
    # DATA LOADER
    # ========================================================

    # num_workers=0 is intentional because your
    # environment is Python 3.14 + WSL.

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    # ========================================================
    # CREATE MOBILENETV2
    # ========================================================

    print(
        "\nLoading MobileNetV2 architecture..."
    )


    model = models.mobilenet_v2(
        weights=None
    )


    # ========================================================
    # REPLACE CLASSIFIER
    # ========================================================

    input_features = (
        model.classifier[1].in_features
    )


    model.classifier[1] = nn.Linear(
        input_features,
        num_classes
    )


    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    print(
        "Loading trained model..."
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )


    # The saved model contains model_state_dict.

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            # Compatibility with a model file
            # containing only state_dict.

            model.load_state_dict(
                checkpoint
            )

    else:

        raise ValueError(
            "Invalid model checkpoint."
        )


    model = model.to(device)

    model.eval()


    print(
        "Model loaded successfully."
    )


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "STARTING TEST EVALUATION"
    )

    print("=" * 60)


    all_predictions = []

    all_labels = []


    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            # Forward pass

            outputs = model(
                images
            )


            # Get predicted class

            _, predictions = torch.max(
                outputs,
                1
            )


            # Store results

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


            # Accuracy

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )


    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    print("\n" + "=" * 60)

    print(
        "FINAL TEST RESULT"
    )

    print("=" * 60)


    print(
        f"\nCorrect predictions : "
        f"{correct}/{total}"
    )


    print(
        f"Test Accuracy      : "
        f"{test_accuracy * 100:.2f}%"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 60)


    report = classification_report(

        all_labels,

        all_predictions,

        labels=np.arange(num_classes),

        target_names=class_names,

        zero_division=0
    )


    print("\n" + report)


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        all_labels,

        all_predictions,

        labels=np.arange(num_classes)
    )


    print("=" * 60)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 60)


    print()


    # Print header

    print(
        f"{'Actual / Predicted':<20}",
        end=""
    )


    for name in class_names:

        print(
            f"{name[:12]:>14}",
            end=""
        )


    print()


    for i, row in enumerate(cm):

        print(
            f"{class_names[i]:<20}",
            end=""
        )


        for value in row:

            print(
                f"{value:>14}",
                end=""
            )


        print()


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    plt.figure(
        figsize=(8, 7)
    )


    plt.imshow(
        cm,
        interpolation="nearest"
    )


    plt.title(
        "Fruit Freshness CNN - Confusion Matrix"
    )


    plt.colorbar()


    tick_positions = np.arange(
        num_classes
    )


    plt.xticks(
        tick_positions,
        class_names,
        rotation=45,
        ha="right"
    )


    plt.yticks(
        tick_positions,
        class_names
    )


    plt.xlabel(
        "Predicted Class"
    )


    plt.ylabel(
        "Actual Class"
    )


    # Write numbers inside cells

    for i in range(num_classes):

        for j in range(num_classes):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )


    plt.tight_layout()


    plt.savefig(
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # PER-CLASS ACCURACY
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "PER-CLASS ACCURACY"
    )

    print("=" * 60)


    for i, class_name in enumerate(
        class_names
    ):

        class_total = cm[i].sum()

        class_correct = cm[i, i]


        if class_total > 0:

            class_accuracy = (
                class_correct /
                class_total
            ) * 100

        else:

            class_accuracy = 0


        print(
            f"{class_name:<20} "
            f"{class_correct}/{class_total} "
            f"({class_accuracy:.2f}%)"
        )


    # ========================================================
    # SAVE EVALUATION RESULTS
    # ========================================================

    results = {

        "test_images": int(total),

        "correct_predictions": int(correct),

        "test_accuracy_percent":
            float(test_accuracy * 100),

        "classes": class_names,

        "confusion_matrix":
            cm.tolist()
    }


    with open(
        "evaluation_results.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "EVALUATION COMPLETED"
    )

    print("=" * 60)


    print(
        "\nGenerated files:"
    )

    print(
        "  confusion_matrix.png"
    )

    print(
        "  evaluation_results.json"
    )


    print(
        "\nNext step:"
    )

    print(
        "  python predict_image.py"
    )


    print("=" * 60)


# ============================================================
# SAFE PYTHON ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()