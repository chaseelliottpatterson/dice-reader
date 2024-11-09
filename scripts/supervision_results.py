from ultralytics import YOLO
import supervision as sv
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO('C:/Users/Studio/Documents/GitHub/dice-reader/scripts/runs/obb/yolov8n_obb_2024-11-09_12-03/weights/best.pt')

# Load an image
image_path = "C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov/images/val/DJI_20241005232542_0139_D.JPG"  # Replace with the path to your image
image = cv2.imread(image_path)

results = model(image, conf=0.85)

# Convert YOLO result to Supervision Detections, including oriented bounding boxes
detections = sv.Detections.from_ultralytics(results[0])

# Initialize OrientedBoxAnnotator
oriented_box_annotator = sv.OrientedBoxAnnotator(thickness=2)

# Annotate the image with oriented bounding boxes
annotated_image = oriented_box_annotator.annotate(scene=image.copy(), detections=detections)

# Display the annotated image with matplotlib
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
