from ultralytics import YOLO
import supervision as sv
import cv2
import matplotlib.pyplot as plt
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
    Color(100, 149, 237)     # Blue for "blue dice"
])
print(detections)
# Initialize OrientedBoxAnnotator with the color palette and class-based color lookup
oriented_box_annotator = sv.OrientedBoxAnnotator(color=color_palette, thickness=2, color_lookup=ColorLookup.CLASS)

# Annotate the image with oriented bounding boxes, using the color palette for each class
annotated_image = oriented_box_annotator.annotate(scene=image.copy(), detections=detections)

# Define labels with confidence scores for each detection
labels = [
    f"{model.names[int(class_id)]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

# Draw labels above each bounding box with a background matching the box color
for label, box, class_id in zip(labels, detections.xyxy, detections.class_id):
    x, y = int(box[0]), int(box[1])

    # Get the color for the text background from the color palette based on the class ID
    color = color_palette.by_idx(class_id).as_bgr()

    # Draw larger background for the label text
    font_scale = 6
    font_thickness = 6
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(annotated_image, (x, y - text_height - 15), (x + text_width + 10, y), color, -1)  # Larger filled rectangle with box color

    # Add the label text on top of the colored background with black text and larger font size
    cv2.putText(
        annotated_image, label, (x + 5, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness  # Black text, larger size, and thicker stroke
    )

# Display the annotated image with labels using matplotlib
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
