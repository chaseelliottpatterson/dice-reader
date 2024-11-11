from ultralytics import YOLO
import supervision as sv
import cv2
import os
from supervision.draw.color import Color, ColorPalette
from supervision.annotators.utils import ColorLookup

# Load the trained YOLOv8 model
model = YOLO('C:/Users/Studio/Documents/GitHub/dice-reader/scripts/runs/obb/yolov8n_obb_2024-11-09_12-03/weights/best.pt')

# Load an image
image_path = "C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov/images/val/DJI_20241005232542_0139_D.JPG"
image = cv2.imread(image_path)

# Run inference with a confidence threshold
results = model(image, conf=0.85)

# Convert YOLO result to Supervision Detections, including oriented bounding boxes
detections = sv.Detections.from_ultralytics(results[0])

# Create a color palette with specific colors for each class (e.g., "red dice" and "blue dice")
color_palette = ColorPalette([
    Color(255, 0, 0),      # Red for "red dice"
    Color(100, 149, 237)   # Blue for "blue dice"
])

# Initialize OrientedBoxAnnotator with the color palette and class-based color lookup
oriented_box_annotator = sv.OrientedBoxAnnotator(color=color_palette, thickness=2, color_lookup=ColorLookup.CLASS)

# Annotate the image with oriented bounding boxes, using the color palette for each class
annotated_image = oriented_box_annotator.annotate(scene=image.copy(), detections=detections)

# Define labels with confidence scores for each detection
labels = [
    f"{model.names[int(class_id)]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Crop and save each detection with a buffer
buffer = 10  # Buffer size in pixels
for i, (box, label) in enumerate(zip(detections.xyxy, labels)):
    x_min, y_min, x_max, y_max = map(int, box)  # Get bounding box coordinates and convert to int
    
    # Apply buffer while ensuring coordinates are within image bounds
    x_min = max(0, x_min - buffer)
    y_min = max(0, y_min - buffer)
    x_max = min(image.shape[1], x_max + buffer)
    y_max = min(image.shape[0], y_max + buffer)
    
    # Crop the ROI from the original image with buffer
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Save the cropped detection
    output_path = os.path.join(output_dir, f"detection_{i+1}_{label.replace(' ', '_')}.jpg")
    cv2.imwrite(output_path, cropped_image)
    print(f"Saved: {output_path}")
