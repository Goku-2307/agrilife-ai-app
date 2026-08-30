import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os


# ============================================================
# SHELFLIFE INTELLIGENCE - REAL-TIME WEBCAM PREDICTION
# ============================================================

MODEL_PATH = "best_fruit_quality_model.pth"
CLASS_NAMES_PATH = "class_names.json"

IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.40


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("       SHELFLIFE INTELLIGENCE - WEBCAM PREDICTION")
print("=" * 60)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"\n{CLASS_NAMES_PATH} not found.\n"
        "Run train.py first."
    )

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(class_names)
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n{MODEL_PATH} not found.\n"
        "Run train.py first."
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
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
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_class_name(class_name):

    parts = class_name.split("_")

    if len(parts) >= 2:
        condition = parts[0].capitalize()
        crop = "_".join(parts[1:]).capitalize()
    else:
        condition = class_name.capitalize()
        crop = "Unknown"

    return crop, condition


def predict_frame(frame):

    # OpenCV uses BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    image = Image.fromarray(rgb)

    # Transform
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to GPU/CPU
    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    confidence = confidence.item()
    predicted_index = predicted_index.item()

    predicted_class = class_names[predicted_index]

    crop, condition = split_class_name(predicted_class)

    return (
        predicted_class,
        crop,
        condition,
        confidence,
        probabilities[0].cpu().numpy()
    )


# ============================================================
# OPEN WEBCAM
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")

    print("\nTry:")
    print("  python -c \"import cv2; c=cv2.VideoCapture(0); print(c.isOpened())\"")

    print("\nIf you are using WSL, webcam access may require")
    print("Windows/WSL camera configuration.")

    raise SystemExit


# Set webcam resolution

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("\n============================================================")
print("WEBCAM STARTED")
print("============================================================")
print("\nShow an apple or banana in front of the camera.")
print("Press Q to quit.")
print("Press S to save the current frame.")


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("\nFailed to read webcam frame.")
        break


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        (
            predicted_class,
            crop,
            condition,
            confidence,
            probabilities
        ) = predict_frame(frame)

    except Exception as e:

        print(f"\nPrediction error: {e}")
        break


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    confidence_percent = confidence * 100


    if confidence >= CONFIDENCE_THRESHOLD:

        status_text = f"{crop} - {condition}"

    else:

        status_text = "UNCERTAIN"


    # Main prediction

    cv2.rectangle(
        frame,
        (10, 10),
        (600, 115),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "ShelfLife Intelligence",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Prediction: {status_text}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence_percent:.2f}%",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # PROBABILITY DISPLAY
    # --------------------------------------------------------

    y_position = 150

    for i, class_name in enumerate(class_names):

        probability = probabilities[i] * 100

        text = f"{class_name}: {probability:.1f}%"

        cv2.putText(
            frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y_position += 30


    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Press Q = Quit | S = Save Image",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # Show frame

    cv2.imshow(
        "ShelfLife Intelligence - Webcam",
        frame
    )


    # --------------------------------------------------------
    # KEYBOARD INPUT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # Quit

    if key == ord("q"):

        break


    # Save image

    elif key == ord("s"):

        filename = "webcam_capture.jpg"

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"\nSaved webcam image: {filename}"
        )

        print(
            f"Prediction: {predicted_class}"
        )

        print(
            f"Confidence: {confidence_percent:.2f}%"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n============================================================")
print("WEBCAM PREDICTION STOPPED")
print("============================================================")