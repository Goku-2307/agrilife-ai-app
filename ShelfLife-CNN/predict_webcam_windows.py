import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os


# ============================================================
# SHELFLIFE INTELLIGENCE - WINDOWS WEBCAM PREDICTION
# ============================================================

MODEL_PATH = "best_fruit_quality_model.pth"
CLASS_NAMES_PATH = "class_names.json"

# Image preprocessing must match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("=" * 60)
print("       SHELFLIFE INTELLIGENCE - WEBCAM PREDICTION")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("Classes:")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print()
print("Loading MobileNetV2...")

model = models.mobilenet_v2(weights=None)

num_classes = len(class_names)

model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n\n"
        "Make sure best_fruit_quality_model.pth is in the "
        "same folder as this script."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# OPEN WINDOWS WEBCAM
# ============================================================

print()
print("Opening Windows webcam...")
print()
print("Press Q to quit.")
print("Press S to save a prediction image.")
print()

# CAP_DSHOW works well with Windows webcams
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():

    # Try normal Windows backend
    camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("=" * 60)
    print("ERROR: Could not open webcam.")
    print("=" * 60)

    print()
    print("Possible solutions:")
    print("1. Check Windows camera permission.")
    print("2. Close Camera/Teams/Zoom if using the webcam.")
    print("3. Make sure your laptop camera is enabled.")
    print("4. Try changing camera index from 0 to 1.")

    raise SystemExit


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read frame.")
        break

    # Convert OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL
    pil_image = Image.fromarray(rgb_frame)

    # Apply preprocessing
    input_tensor = transform(pil_image)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # --------------------------------------------------------
    # CNN prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    class_index = predicted.item()

    confidence_value = confidence.item() * 100

    predicted_class = class_names[class_index]

    # --------------------------------------------------------
    # Extract crop and condition
    # --------------------------------------------------------

    parts = predicted_class.split("_")

    if len(parts) >= 2:

        condition = parts[0].capitalize()

        crop = "_".join(parts[1:]).capitalize()

    else:

        condition = predicted_class
        crop = "Unknown"


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    label = (
        f"{crop} - {condition} "
        f"({confidence_value:.1f}%)"
    )

    # Main prediction
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # CNN label
    cv2.putText(
        frame,
        f"Class: {predicted_class}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Instruction
    cv2.putText(
        frame,
        "Q = Quit | S = Save",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show frame
    cv2.imshow(
        "ShelfLife Intelligence - CNN",
        frame
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):

        break

    # Save current frame
    elif key == ord("s"):

        filename = "webcam_prediction.jpg"

        cv2.imwrite(
            filename,
            frame
        )

        print()
        print("Prediction saved:")
        print(filename)
        print()
        print("Crop:", crop)
        print("Condition:", condition)
        print(
            f"Confidence: {confidence_value:.2f}%"
        )


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("=" * 60)
print("WEBCAM PREDICTION CLOSED")
print("=" * 60)