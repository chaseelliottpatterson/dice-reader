from ultralytics import YOLO  
import os
import datetime

def main():
   # Load a pre-trained YOLO model for classification
   model = YOLO('yolov8n-cls.pt')  # Use the correct path and name for your model
   save_dir = 'C:/Users/Studio/Documents/GitHub/dice-reader'
   os.makedirs(save_dir, exist_ok=True)
   timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")

   model.train(
      data='C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-class-cropped-organized',  # Update this if you have a classification-specific data file
      epochs=100,               # Adjust as needed
      batch=32,                 # Larger batch size may work for classification
      # imgsz=224,                # Common image size for classification (YOLO supports flexible sizing)
      lr0=0.001,                # Learning rate for classification
      # optimizer='AdamW',        # Optimizer
      weight_decay=0.0001,      # Regularization
      patience=20,              # Early stopping patience
      device=0,
      name=f'yolov8n_cls_{timestamp}'
   )

if __name__ == "__main__":
    main()