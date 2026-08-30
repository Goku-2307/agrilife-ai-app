import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# SHELFLIFE INTELLIGENCE - VISUAL IMAGE PREDICTION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "best_fruit_quality_model.pth"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.json"

# Change this whenever you want to test another image
IMAGE_PATH = PROJECT_DIR / "test.jpg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return [data[str(i)] for i in range(len(data))]


# ============================================================
# BUILD MOBILE NET V2
# ============================================================

def build_model(num_classes):

    print("\nLoading MobileNetV2...")

    model = models.mobilenet_v2(weights=None)

    # Replace ImageNet classifier with our 4-class classifier
    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Support different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("       SHELFLIFE INTELLIGENCE - VISUAL PREDICTION")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():

        print("\nImage not found:")
        print(IMAGE_PATH)

        print("\nChange IMAGE_PATH at the top of this file.")

        return

    # --------------------------------------------------------
    # LOAD CLASSES
    # --------------------------------------------------------

    class_names = load_class_names()

    print("\nClasses:")

    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # --------------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print(f"\nLoading image: {IMAGE_PATH.name}")

    image = Image.open(IMAGE_PATH).convert("RGB")

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = build_model(len(class_names))

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nRunning prediction...")

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    # Highest probability
    confidence, predicted_index = torch.max(
        probabilities,
        dim=0
    )

    predicted_class = class_names[
        predicted_index.item()
    ]

    # --------------------------------------------------------
    # EXTRACT CROP + CONDITION
    # --------------------------------------------------------

    parts = predicted_class.split("_", 1)

    condition = parts[0].capitalize()

    if len(parts) > 1:
        crop = parts[1].capitalize()
    else:
        crop = predicted_class

    # --------------------------------------------------------
    # TERMINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("                 PREDICTION RESULT")
    print("=" * 60)

    print(f"\nImage              : {IMAGE_PATH.name}")

    print(f"Crop               : {crop}")

    print(f"Visual Condition   : {condition}")

    print(f"Predicted Class    : {predicted_class}")

    print(
        f"Confidence         : "
        f"{confidence.item() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # ALL PROBABILITIES
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        probabilities,
        descending=True
    )

    print("\n" + "=" * 60)
    print("                CLASS PROBABILITIES")
    print("=" * 60)

    for index in sorted_indices:

        name = class_names[index.item()]

        probability = (
            probabilities[index].item() * 100
        )

        print(
            f"{name:<22} "
            f"{probability:6.2f}%"
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    names = [
        class_names[i.item()]
        for i in sorted_indices
    ]

    values = [
        probabilities[i].item() * 100
        for i in sorted_indices
    ]

    # Create figure
    fig = plt.figure(figsize=(12, 6))

    # --------------------------------------------------------
    # LEFT: ORIGINAL IMAGE
    # --------------------------------------------------------

    ax1 = fig.add_axes(
        [0.03, 0.15, 0.45, 0.75]
    )

    ax1.imshow(image)

    ax1.axis("off")

    ax1.set_title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence.item() * 100:.2f}%",
        fontsize=13
    )

    # --------------------------------------------------------
    # RIGHT: PROBABILITY GRAPH
    # --------------------------------------------------------

    ax2 = fig.add_axes(
        [0.55, 0.18, 0.40, 0.68]
    )

    ax2.barh(
        names[::-1],
        values[::-1]
    )

    ax2.set_xlim(0, 100)

    ax2.set_xlabel(
        "Probability (%)"
    )

    ax2.set_title(
        "MobileNetV2 Class Probabilities"
    )

    ax2.grid(
        axis="x",
        alpha=0.25
    )

    # Add percentage values
    for y, value in enumerate(values[::-1]):

        ax2.text(
            min(value + 1, 98),
            y,
            f"{value:.2f}%",
            va="center"
        )

    plt.suptitle(
        "ShelfLife Intelligence - CNN Image Prediction",
        fontsize=16
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    output_path = (
        PROJECT_DIR /
        "prediction_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\n" + "=" * 60)
    print("VISUALIZATION SAVED")
    print("=" * 60)

    print(f"\nSaved as:")
    print(output_path)

    print("\nClose the graph window to finish.")

    plt.show()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()