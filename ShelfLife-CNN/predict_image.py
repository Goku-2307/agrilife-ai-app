import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# SHELFLIFE INTELLIGENCE - SINGLE IMAGE PREDICTION
# ============================================================

MODEL_PATH = "best_fruit_quality_model.pth"
CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = 224

import sys

# Default image path (can also be passed as command-line argument)
IMAGE_PATH = "apple.jpg"


# ============================================================
# MAIN
# ============================================================

def main():
    global IMAGE_PATH

    print("=" * 60)
    print("       SHELFLIFE INTELLIGENCE - IMAGE PREDICTION")
    print("=" * 60)

    # Check if image passed as CLI argument
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]

    # Fallback search if default image not found
    if not Path(IMAGE_PATH).exists():
        fallbacks = [
            Path("../sample_images/fresh_apple.jpg"),
            Path("sample_images/fresh_apple.jpg"),
            Path("../sample_images/fresh_banana.jpg"),
        ]
        for fb in fallbacks:
            if fb.exists():
                print(f"[INFO] '{IMAGE_PATH}' not found. Falling back to sample image: {fb}")
                IMAGE_PATH = str(fb)
                break

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

    if not Path(MODEL_PATH).exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )


    if not Path(CLASS_NAMES_PATH).exists():

        raise FileNotFoundError(
            f"\nClass names file not found:\n"
            f"{CLASS_NAMES_PATH}"
        )


    if not Path(IMAGE_PATH).exists():

        raise FileNotFoundError(
            f"\nImage not found: {IMAGE_PATH}\n\n"
            f"Usage: python predict_image.py <path_to_image>\n"
            f"Example: python predict_image.py ../sample_images/fresh_apple.jpg"
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
    # IMAGE TRANSFORM
    # ========================================================

    transform = transforms.Compose([

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
    # LOAD IMAGE
    # ========================================================

    print(
        f"\nLoading image: {IMAGE_PATH}"
    )


    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")


    # Keep original image for processing
    # and create transformed tensor.

    image_tensor = transform(
        image
    )


    # Add batch dimension

    image_tensor = image_tensor.unsqueeze(0)


    image_tensor = image_tensor.to(
        device
    )


    # ========================================================
    # CREATE MOBILENETV2
    # ========================================================

    print(
        "\nLoading MobileNetV2..."
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

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )


    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )


    model = model.to(device)

    model.eval()


    print(
        "Model loaded successfully."
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\nRunning prediction..."
    )


    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        # Convert model output to probabilities

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Get highest probability

        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    predicted_index = (
        predicted_index.item()
    )


    confidence = (
        confidence.item() * 100
    )


    predicted_class = (
        class_names[predicted_index]
    )


    # ========================================================
    # EXTRACT FRUIT + CONDITION
    # ========================================================

    # Example:
    #
    # fresh_apple
    # rotten_banana

    parts = predicted_class.split(
        "_",
        1
    )


    if len(parts) == 2:

        condition = parts[0]

        fruit = parts[1]

    else:

        condition = "Unknown"

        fruit = predicted_class


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "             PREDICTION RESULT"
    )

    print("=" * 60)


    print(
        f"\nImage              : {IMAGE_PATH}"
    )


    print(
        f"Crop               : "
        f"{fruit.capitalize()}"
    )


    print(
        f"Visual Condition   : "
        f"{condition.capitalize()}"
    )


    print(
        f"Predicted Class    : "
        f"{predicted_class}"
    )


    print(
        f"Confidence         : "
        f"{confidence:.2f}%"
    )


    print("\n" + "=" * 60)


    # ========================================================
    # SHOW ALL CLASS PROBABILITIES
    # ========================================================

    print(
        "CLASS PROBABILITIES"
    )

    print("=" * 60)


    probability_values = (
        probabilities[0]
        .cpu()
        .numpy()
    )


    # Sort from highest to lowest

    sorted_indices = np.argsort(
        probability_values
    )[::-1]


    for index in sorted_indices:

        probability = (
            probability_values[index]
            * 100
        )


        print(
            f"{class_names[index]:<20}"
            f"{probability:>8.2f}%"
        )


    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Import numpy here so the program remains
    # simple and lightweight.

    import numpy as np

    main()