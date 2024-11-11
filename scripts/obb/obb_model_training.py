from ultralytics import YOLO
import os
import datetime

def main():
   model = YOLO('C:/Users/Studio/Documents/GitHub/dice-reader/scripts/models/yolo11n-obb.pt')
   save_dir = 'C:/Users/Studio/Documents/GitHub/dice-reader'
   os.makedirs(save_dir, exist_ok=True)
   timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")

   model.train(
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

if __name__ == "__main__":
    main()
