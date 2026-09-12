from ultralytics import YOLO
import os
from utils.config import YOLO_MODEL_CLASSIFICATION, EPOCHS, IMG_SIZE, BATCH_SIZE

def train_classifier(data_path: str):
    """Train YOLOv8 for image classification."""
    model = YOLO(YOLO_MODEL_CLASSIFICATION)
    
    print(f"Starting classification training on dataset: {data_path}")
    results = model.train(
        data=data_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project='models',
        name='classifier_run'
    )
    print("Training complete. Models saved to models/classifier_run/")

if __name__ == "__main__":
    dataset_dir = "data/processed"
    if os.path.exists(dataset_dir):
        train_classifier(dataset_dir)
    else:
        print("Dataset not found. Please run prepare_dataset.py first.")
