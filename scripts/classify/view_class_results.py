import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# Define the base path for the images
base_path = "C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-class-cropped-organized/test"

# Function to randomly select an image file from the directory structure
def get_random_image_path(base_path):
    image_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))
    return random.choice(image_files) if image_files else None

# Get a random image path
image_path = get_random_image_path(base_path)
if image_path is None:
    raise FileNotFoundError("No image files found in the specified directory.")

# Load the trained YOLOv8 classification model
model = YOLO('C:/Users/Studio/Documents/GitHub/dice-reader/scripts/classify/runs/classify/yolov8n_cls_2024_11_11__14_49/weights/best.pt')

# Load the random image for classification
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

# Convert image to RGB format, normalize, and resize if needed
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1] range
image_resized = cv2.resize(image_rgb, (224, 224))  # Adjust size if necessary for your model

# Convert image to torch.Tensor and add batch dimension
image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)  # Convert HWC to CHW format

# Run inference with the model
results = model(image_tensor)

# Check if the results contain probabilities
if hasattr(results[0], 'probs') and results[0].probs is not None:
    # Access the top-1 prediction and confidence
    class_id = results[0].probs.top1
    confidence = results[0].probs.top1conf.item()  # Convert to a Python float

    # Prepare the label text with the class name and confidence
    label = f"{model.names[class_id]} {confidence:.2f}"

    # Annotate the image with the label at the top-left corner
    font_scale = 2
    font_thickness = 2
    (x, y) = (10, 50)
    cv2.putText(
        image, label, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness  # Red text for the label
    )

    # Display the annotated image with the label using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("Error: No classification probabilities found in the model output.")
