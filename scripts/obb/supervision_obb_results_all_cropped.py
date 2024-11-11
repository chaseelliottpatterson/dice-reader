from ultralytics import YOLO
import supervision as sv
import cv2
import os
import shutil
from supervision.draw.color import Color, ColorPalette
from supervision.annotators.utils import ColorLookup

# Paths
INPUT_DIR = "C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov"
OUTPUT_DIR = "C:/Users/Studio/Documents/GitHub/dice-reader/output"
model_path = 'C:/Users/Studio/Documents/GitHub/dice-reader/scripts/runs/obb/yolov8n_obb_2024-11-09_12-03/weights/best.pt'

# Load the trained YOLOv8 model
model = YOLO(model_path)

# Set up color palette for annotations
color_palette = ColorPalette([
    Color(255, 0, 0),      # Red for "red dice"
    Color(100, 149, 237)   # Blue for "blue dice"
])

# OrientedBoxAnnotator setup
oriented_box_annotator = sv.OrientedBoxAnnotator(color=color_palette, thickness=2, color_lookup=ColorLookup.CLASS)

# Buffer size for cropping
buffer = 8

# Clear OUTPUT_DIR if it exists and recreate it
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Traverse through all subdirectories and process each image file
for root, _, files in os.walk(INPUT_DIR):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)

            # Run inference
            results = model(image, conf=0.7)
            
            # Convert YOLO result to Supervision Detections
            detections = sv.Detections.from_ultralytics(results[0])

            # Annotate the image
            annotated_image = oriented_box_annotator.annotate(scene=image.copy(), detections=detections)

            # Define labels with confidence scores for each detection
            labels = [
                f"{model.names[int(class_id)]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]

            # Crop and save each detection with a buffer
            for i, (box, label) in enumerate(zip(detections.xyxy, labels)):
                x_min, y_min, x_max, y_max = map(int, box)  # Bounding box coordinates

                # Apply buffer while ensuring coordinates are within image bounds
                x_min = max(0, x_min - buffer)
                y_min = max(0, y_min - buffer)
                x_max = min(image.shape[1], x_max + buffer)
                y_max = min(image.shape[0], y_max + buffer)

                # Crop the ROI from the original image with buffer
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Save the cropped detection directly in the OUTPUT_DIR
                output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_detection_{i+1}.jpg")
                cv2.imwrite(output_path, cropped_image)
                print(f"Saved: {output_path}")
