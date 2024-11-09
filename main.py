from ultralytics import YOLO
 
# Load the model.
model = YOLO('models/yolo11n-obb.pt')
 
# Training.
results = model.train(
   data='data/dice-reader_test_set_cov/data.yaml',
   epochs=50,
   batch=8,
   name='yolov8n_v8_obb_50e'
)
print("done")