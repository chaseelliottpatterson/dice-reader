import os
import random
from ultralytics import YOLO
import supervision as sv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Define model paths
obb_model_path = 'C:/Users/Studio/Documents/GitHub/dice-reader/scripts/obb/runs/obb/yolov8n_obb_2024-11-09_12-03/weights/best.pt'
class_model_path = 'C:/Users/Studio/Documents/GitHub/dice-reader/scripts/classify/runs/classify/yolov8n_cls_2024_11_11__14_49/weights/best.pt'

# Define image directory
image_dir = 'C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov/images/'

# Load models
obb_model = YOLO(obb_model_path)
class_model = YOLO(class_model_path)

# Select a random image from the directory tree
def select_random_image(directory):
    all_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".JPG"):
                all_images.append(os.path.join(root, file))
    return random.choice(all_images) if all_images else None

# Display original and cropped images in a grid
def display_images(original, crops):
    num_images = len(crops) + 1
    cols = 3
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))

    # Display the original image
    plt.subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Display each cropped detection
    for i, crop in enumerate(crops):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title(f"Cropped Detection {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Process image: Detect, crop, classify, and display results
def process_dice_image(image_path):
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

    # Run OBB model
    results = obb_model(image, conf=0.85)
    detections = sv.Detections.from_ultralytics(results[0])

    # List to hold cropped images and results
    cropped_images = []
    dice_info = []

    for i, box in enumerate(detections.xyxy):
        # Add 12-pixel buffer to the bounding box
        x1, y1, x2, y2 = box
        x1, y1 = max(int(x1 - 12), 0), max(int(y1 - 12), 0)
        x2, y2 = min(int(x2 + 12), image.shape[1]), min(int(y2 + 12), image.shape[0])

        # Crop the detected dice area
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)

        # Prepare cropped image for classification
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB) / 255.0
        cropped_image_resized = cv2.resize(cropped_image_rgb, (224, 224))
        cropped_image_tensor = torch.from_numpy(cropped_image_resized).float().permute(2, 0, 1).unsqueeze(0)

        # Run classification model
        class_results = class_model(cropped_image_tensor)

        # Get class ID and confidence
        if hasattr(class_results[0], 'probs') and class_results[0].probs is not None:
            class_id = class_results[0].probs.top1
            confidence = class_results[0].probs.top1conf.item()

            # Assign color based on class_id in OBB detections
            color = "red" if detections.class_id[i] == 0 else "blue"  # Modify as per your classes

            # Append result
            dice_info.append((class_model.names[class_id], confidence, color))

            print(f"Dice {i+1}: Number: {class_model.names[class_id]}, Confidence: {confidence:.2f}, Color: {color}")
        
        else:
            print(f"Error: No classification probabilities found for Dice {i+1}.")

    # Display the original and all cropped images together in a grid
    display_images(image, cropped_images)
    return dice_info

# Run the process on a randomly selected image
random_image_path = select_random_image(image_dir)
if random_image_path:
    print(f"Processing random image: {random_image_path}")
    process_dice_image(random_image_path)
else:
    print("No images found in the specified directory.")
