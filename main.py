from ultralytics import YOLO
 
# Load the model.
model = YOLO('models/yolo11n-obb.pt')
 
# # Training.
# results = model.train(
#    data='data/dice-reader_test_set_cov/data.yaml',
#    imgsz=640,
#    epochs=50,
#    batch=8,
#    name='yolov8n_v8_obb_50e'
# )

results = model.train(
   data='data/dice-reader_test_set_cov/data.yaml',
    epochs=100,                 # Increase if needed
    batch=16,                   # Adjust based on GPU capacity
    imgsz=640,                  # Resolution of input images
    lr0=0.002,                  # Initial learning rate
    optimizer='AdamW',          # Optimizer
    weight_decay=0.0005,        # Regularization to prevent overfitting
    augment=True,                # Enable data augmentation
    device=0,
    name='yolov8n_v8_obb_50e'
)
print("done")