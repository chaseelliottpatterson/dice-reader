from ultralytics import YOLO
import datetime

def main():
   timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
   model = YOLO('models/yolo11n-obb.pt')
   
   # # Training debugging
   # results = model.train(
   #    data='../data/dice-reader_test_set_cov/data.yaml',
   #    imgsz=640,
   #    epochs=50,
   #    batch=8,
   #    # amp=False,
   #    # device='cpu',
   #    name='yolov8n_v8_obb_50e'

   # )

   results = model.train(
      data='C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov/data.yaml',
      epochs=300,                 # Increase if needed
      batch=16,                   # Adjust based on GPU capacity
      imgsz=1024,                  # Resolution of input images
      lr0=0.002,                  # Initial learning rate
      optimizer='AdamW',          # Optimizer
      weight_decay=0.0005,        # Regularization to prevent overfitting
      patience = 30,
      device=0,
      time=1,
      name=f'yolov8n_obb_{timestamp}'
   )
   # results = model.train(
   #    data='C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-reader-139_cov/data.yaml',
   #    # data='../data/dice-dice-reader-139_cov/data.yaml',
   #    epochs=300,                 # Increase if needed
   #    batch=32,                   # Adjust based on GPU capacity
   #    device=0,
   #    name=f'yolov8n_obb_{timestamp}'
   # )

   print("done")
if __name__ == "__main__":
    main()
