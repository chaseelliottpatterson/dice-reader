from ultralytics import YOLO
import supervision as sv
import cv2
import torch
import numpy as np

# Paths to the models
obb_model_path = 'C:/Users/Studio/Documents/GitHub/dice-reader/scripts/obb/runs/obb/yolov8n_obb_2024-11-09_12-03/weights/best.pt'
class_model_path = 'C:/Users/Studio/Documents/GitHub/dice-reader/scripts/classify/runs/classify/yolov8n_cls_2024_11_11__14_49/weights/best.pt'

# Load the models
obb_model = YOLO(obb_model_path)
class_model = YOLO(class_model_path)

# Function to run the OBB model and crop detections
def process_dice(image_path):
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")
    
    # Run the OBB model
    results = obb_model(image, conf=0.85)
    detections = sv.Detections.from_ultralytics(results[0])
    
    dice_info = []

    for i, box in enumerate(detections.xyxy):
        # Add a 12-pixel buffer to the bounding box
        x1, y1, x2, y2 = box
        x1, y1 = max(int(x1 - 12), 0), max(int(y1 - 12), 0)
        x2, y2 = min(int(x2 + 12), image.shape[1]), min(int(y2 + 12), image.shape[0])

        # Crop the detected dice area
        cropped_image = image[y1:y2, x1:x2]

        # Convert the cropped image for classification
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB) / 255.0
        cropped_image_resized = cv2.resize(cropped_image_rgb, (224, 224))
        cropped_image_tensor = torch.from_numpy(cropped_image_resized).float().permute(2, 0, 1).unsqueeze(0)

        # Run the classification model
        class_results = class_model(cropped_image_tensor)
        
        # Get the class (number) and confidence
        if hasattr(class_results[0], 'probs') and class_results[0].probs is not None:
            class_id = class_results[0].probs.top1
            confidence = class_results[0].probs.top1conf.item()

            # Record the color based on OBB detection class
            color = "red" if detections.class_id[i] == 0 else "blue"  # Modify as per your classes

            # Save the result
            dice_info.append((class_model.names[class_id], confidence, color))

            print(f"Dice {i+1}: Number: {class_model.names[class_id]}, Confidence: {confidence:.2f}, Color: {color}")
        
        else:
            print(f"Error: No classification probabilities found for Dice {i+1}.")

    return dice_info

# Run the function on your specified image
image_path = 'C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov/images/test/DJI_20241005233639_0181_D.JPG'
dice_info = process_dice(image_path)
