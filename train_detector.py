from ultralytics import YOLO
from utils.config import YOLO_MODEL_DETECTION, EPOCHS, IMG_SIZE, BATCH_SIZE

def train_detector(data_yaml: str):
    """Train YOLOv8 for object detection."""
    model = YOLO(YOLO_MODEL_DETECTION)
    
    print(f"Starting detection training using config: {data_yaml}")
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project='models',
        name='detector_run'
    )
    print("Training complete. Models saved to models/detector_run/")

if __name__ == "__main__":
    train_detector("data/dataset.yaml")
