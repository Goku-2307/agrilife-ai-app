import json
import copy
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SHELFLIFE INTELLIGENCE - CNN TRAINING
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_EPOCHS = 25

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

MODEL_PATH = "best_fruit_quality_model.pth"

CLASS_NAMES_PATH = "class_names.json"

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 7


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("       SHELFLIFE INTELLIGENCE - CNN TRAINING")
    print("=" * 60)


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )


    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )


    train_path = DATASET_DIR / "train"

    val_path = DATASET_DIR / "val"

    test_path = DATASET_DIR / "test"


    if not train_path.exists():

        raise FileNotFoundError(
            f"Training folder not found: {train_path}"
        )

    if not val_path.exists():

        raise FileNotFoundError(
            f"Validation folder not found: {val_path}"
        )

    if not test_path.exists():

        raise FileNotFoundError(
            f"Testing folder not found: {test_path}"
        )


    # ========================================================
    # DATA TRANSFORMS
    # ========================================================

    # Training images receive augmentation.
    #
    # This helps the model handle:
    # - different angles
    # - lighting
    # - rotation
    # - distance
    # - small changes in appearance

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.80, 1.0)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
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


    # Validation and test images are not augmented.

    val_test_transform = transforms.Compose([

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
    # LOAD DATASETS
    # ========================================================

    print("\nLoading datasets...")


    train_dataset = datasets.ImageFolder(
        train_path,
        transform=train_transform
    )


    val_dataset = datasets.ImageFolder(
        val_path,
        transform=val_test_transform
    )


    test_dataset = datasets.ImageFolder(
        test_path,
        transform=val_test_transform
    )


    # ========================================================
    # CLASS INFORMATION
    # ========================================================

    class_names = train_dataset.classes

    num_classes = len(class_names)


    print("\nClasses:")

    for index, class_name in enumerate(class_names):

        print(
            f"  {index}: {class_name}"
        )


    print(
        f"\nNumber of classes: {num_classes}"
    )


    # ========================================================
    # CHECK CLASS CONSISTENCY
    # ========================================================

    if val_dataset.classes != class_names:

        raise ValueError(
            "\nERROR: Validation classes do not "
            "match training classes."
        )


    if test_dataset.classes != class_names:

        raise ValueError(
            "\nERROR: Test classes do not "
            "match training classes."
        )


    # ========================================================
    # SAVE CLASS NAMES
    # ========================================================

    with open(
        CLASS_NAMES_PATH,
        "w"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )


    print(
        f"\nClass names saved to: "
        f"{CLASS_NAMES_PATH}"
    )


    # ========================================================
    # DATASET SIZES
    # ========================================================

    print("\nDataset sizes:")

    print(
        f"Training   : {len(train_dataset)}"
    )

    print(
        f"Validation : {len(val_dataset)}"
    )

    print(
        f"Testing    : {len(test_dataset)}"
    )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print(
        "\nTraining class distribution:"
    )


    class_counts = []


    for class_index, class_name in enumerate(
        class_names
    ):

        count = sum(
            1
            for _, label in train_dataset.samples
            if label == class_index
        )

        class_counts.append(count)

        print(
            f"  {class_name:<15}: {count}"
        )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    # Your dataset is imbalanced:
    #
    # fresh_apple  = 86
    # fresh_banana = 13
    # rotten_apple = 38
    # rotten_banana = 62
    #
    # Class weights prevent the model from
    # ignoring the smaller classes.


    classes_for_weights = np.arange(
        num_classes
    )


    targets_array = np.array(
        train_dataset.targets
    )


    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_for_weights,
        y=targets_array
    )


    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)


    print("\nClass weights:")


    for name, weight in zip(
        class_names,
        class_weights
    ):

        print(
            f"  {name:<15}: "
            f"{weight.item():.4f}"
        )


    # ========================================================
    # DATA LOADERS
    # ========================================================

    # IMPORTANT:
    #
    # num_workers=0 is intentional.
    #
    # Your system is:
    # Python 3.14
    # Ubuntu 26.04
    # WSL
    #
    # Multiple DataLoader workers were causing
    # the multiprocessing error.
    #
    # With only 199 training images, num_workers=0
    # is completely fine.


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    # ========================================================
    # LOAD MOBILENETV2
    # ========================================================

    print(
        "\nLoading pretrained MobileNetV2..."
    )


    weights = models.MobileNet_V2_Weights.DEFAULT


    model = models.mobilenet_v2(
        weights=weights
    )


    # ========================================================
    # FREEZE FEATURE EXTRACTOR
    # ========================================================

    # MobileNetV2 already learned general
    # visual features from a large dataset.
    #
    # We reuse those features and train
    # our own fruit classifier.


    for parameter in model.features.parameters():

        parameter.requires_grad = False


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
    # MOVE MODEL TO GPU
    # ========================================================

    model = model.to(device)


    print(
        "MobileNetV2 loaded successfully."
    )


    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3
    )


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    training_losses = []

    validation_losses = []

    training_accuracies = []

    validation_accuracies = []


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_val_accuracy = 0.0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0


    # ========================================================
    # TRAINING FUNCTION
    # ========================================================

    def train_one_epoch():

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            # Clear gradients

            optimizer.zero_grad()


            # Forward pass

            outputs = model(
                images
            )


            # Calculate loss

            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation

            loss.backward()


            # Update model

            optimizer.step()


            # Loss

            running_loss += (
                loss.item()
                * images.size(0)
            )


            # Predictions

            _, predictions = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predictions == labels
            ).sum().item()


        epoch_loss = (
            running_loss / total
        )


        epoch_accuracy = (
            correct / total
        ) * 100


        return (
            epoch_loss,
            epoch_accuracy
        )


    # ========================================================
    # VALIDATION FUNCTION
    # ========================================================

    def validate():

        model.eval()

        running_loss = 0.0

        correct = 0

        total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )


                outputs = model(
                    images
                )


                loss = criterion(
                    outputs,
                    labels
                )


                running_loss += (
                    loss.item()
                    * images.size(0)
                )


                _, predictions = torch.max(
                    outputs,
                    1
                )


                total += labels.size(0)


                correct += (
                    predictions == labels
                ).sum().item()


        epoch_loss = (
            running_loss / total
        )


        epoch_accuracy = (
            correct / total
        ) * 100


        return (
            epoch_loss,
            epoch_accuracy
        )


    # ========================================================
    # START TRAINING
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "STARTING TRAINING"
    )

    print("=" * 60)


    start_time = time.time()


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        NUM_EPOCHS
    ):


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch()
        )


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate()
        )


        # ----------------------------------------------------
        # UPDATE SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_accuracy
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        training_losses.append(
            train_loss
        )

        validation_losses.append(
            val_loss
        )

        training_accuracies.append(
            train_accuracy
        )

        validation_accuracies.append(
            val_accuracy
        )


        # ----------------------------------------------------
        # PRINT EPOCH
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"[{epoch + 1}/{NUM_EPOCHS}]"
        )


        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )


        print(
            f"Train Accuracy : "
            f"{train_accuracy:.2f}%"
        )


        print(
            f"Val Loss       : "
            f"{val_loss:.4f}"
        )


        print(
            f"Val Accuracy   : "
            f"{val_accuracy:.2f}%"
        )


        print(
            f"Learning Rate  : "
            f"{optimizer.param_groups[0]['lr']:.6f}"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:


            best_val_accuracy = (
                val_accuracy
            )


            best_model_weights = (
                copy.deepcopy(
                    model.state_dict()
                )
            )


            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        class_names,

                    "num_classes":
                        num_classes,

                    "image_size":
                        IMAGE_SIZE,

                    "best_val_accuracy":
                        best_val_accuracy
                },

                MODEL_PATH
            )


            print(
                f"✓ Best model saved "
                f"({best_val_accuracy:.2f}%)"
            )


            epochs_without_improvement = 0


        else:

            epochs_without_improvement += 1


            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break


    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_model_weights
    )


    # ========================================================
    # TRAINING TIME
    # ========================================================

    training_time = (
        time.time() - start_time
    )


    # ========================================================
    # TRAINING COMPLETED
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "TRAINING COMPLETED"
    )

    print("=" * 60)


    print(
        f"\nBest Validation Accuracy: "
        f"{best_val_accuracy:.2f}%"
    )


    print(
        f"Training Time: "
        f"{training_time / 60:.2f} minutes"
    )


    print(
        f"\nModel saved as:"
    )


    print(
        f"  {MODEL_PATH}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    torch.save(

        {
            "model_state_dict":
                model.state_dict(),

            "class_names":
                class_names,

            "num_classes":
                num_classes,

            "image_size":
                IMAGE_SIZE,

            "best_val_accuracy":
                best_val_accuracy
        },

        MODEL_PATH
    )


    # ========================================================
    # TRAINING LOSS GRAPH
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        training_losses,
        label="Training Loss"
    )


    plt.plot(
        validation_losses,
        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Training and Validation Loss"
    )


    plt.legend()


    plt.grid()


    plt.savefig(
        "training_loss.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # ACCURACY GRAPH
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        training_accuracies,
        label="Training Accuracy"
    )


    plt.plot(
        validation_accuracies,
        label="Validation Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy (%)"
    )


    plt.title(
        "Training and Validation Accuracy"
    )


    plt.legend()


    plt.grid()


    plt.savefig(
        "training_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "\nTraining graphs saved:"
    )

    print(
        "  training_loss.png"
    )

    print(
        "  training_accuracy.png"
    )


    print("\n" + "=" * 60)

    print(
        "READY FOR EVALUATION"
    )

    print("=" * 60)


    print(
        "\nNext step:"
    )

    print(
        "    python evaluate.py"
    )

    print("=" * 60)


# ============================================================
# SAFE PYTHON ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()