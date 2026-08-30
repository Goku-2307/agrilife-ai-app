import os
import cv2
import numpy as np

os.makedirs("sample_images", exist_ok=True)

def create_fruit_image(filename: str, fruit_type: str, condition: str):
    img = np.full((400, 400, 3), (240, 240, 240), dtype=np.uint8)
    
    if fruit_type == "apple":
        if condition == "fresh":
            # Bright red glossy apple
            cv2.circle(img, (200, 220), 120, (30, 30, 220), -1)
            cv2.circle(img, (160, 180), 30, (70, 70, 255), -1)  # highlight
            # Stem & leaf
            cv2.rectangle(img, (195, 80), (205, 110), (30, 70, 120), -1)
            cv2.ellipse(img, (225, 95), (25, 10), 30, 0, 360, (34, 139, 34), -1)
        else:
            # Brownish dark rotten apple with spots
            cv2.circle(img, (200, 220), 120, (30, 60, 110), -1)
            cv2.circle(img, (170, 200), 25, (20, 35, 60), -1)
            cv2.circle(img, (230, 240), 20, (15, 30, 50), -1)
            cv2.circle(img, (190, 260), 30, (25, 45, 75), -1)

    elif fruit_type == "banana":
        if condition == "fresh":
            # Vibrant yellow curved banana
            pts = np.array([[120, 280], [180, 320], [280, 280], [320, 180], [280, 150], [240, 250], [160, 260]], np.int32)
            cv2.fillPoly(img, [pts], (20, 220, 255))
            # Green tip
            cv2.circle(img, (120, 280), 15, (40, 180, 50), -1)
        else:
            # Blackened spotted rotten banana
            pts = np.array([[120, 280], [180, 320], [280, 280], [320, 180], [280, 150], [240, 250], [160, 260]], np.int32)
            cv2.fillPoly(img, [pts], (30, 80, 100))
            cv2.circle(img, (200, 280), 18, (15, 20, 25), -1)
            cv2.circle(img, (260, 220), 22, (10, 15, 20), -1)

    elif fruit_type == "tomato":
        # Bright red tomato
        cv2.circle(img, (200, 220), 120, (20, 40, 230), -1)
        # Green calyx
        cv2.circle(img, (200, 100), 10, (30, 160, 30), -1)
        cv2.line(img, (200, 100), (160, 80), (30, 160, 30), 4)
        cv2.line(img, (200, 100), (240, 80), (30, 160, 30), 4)
        cv2.line(img, (200, 100), (200, 60), (30, 160, 30), 4)

    elif fruit_type == "mango":
        # Golden orange mango
        cv2.ellipse(img, (200, 210), (90, 130), 25, 0, 360, (20, 160, 255), -1)
        cv2.circle(img, (200, 80), 8, (30, 120, 40), -1)

    cv2.putText(img, f"{condition.upper()} {fruit_type.upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.imwrite(os.path.join("sample_images", filename), img)
    print(f"Created sample image: sample_images/{filename}")

create_fruit_image("fresh_apple.jpg", "apple", "fresh")
create_fruit_image("rotten_apple.jpg", "apple", "rotten")
create_fruit_image("fresh_banana.jpg", "banana", "fresh")
create_fruit_image("rotten_banana.jpg", "banana", "rotten")
create_fruit_image("fresh_tomato.jpg", "tomato", "fresh")
create_fruit_image("fresh_mango.jpg", "mango", "fresh")
print("Sample images ready!")
