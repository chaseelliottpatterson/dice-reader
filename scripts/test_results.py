from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO('C:/Users/Studio/Documents/GitHub/dice-reader/scripts/runs/obb/yolov8n_obb_2024-11-09_12-03/weights/best.pt')

# Load an image
image_path = "C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov/images/val/DJI_20241005232542_0139_D.JPG"  # Replace with the path to your image
image = cv2.imread(image_path)

# Run inference
results = model(image, conf=0.1)

# Print detection results
print(results)  # This will print bounding box coordinates, classes, and confidence scores

# Draw bounding boxes on the image
results_plotted = results[0].plot()  # Access the first result and plot it

# Display the image with bounding boxes using matplotlib
plt.imshow(cv2.cvtColor(results_plotted, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
