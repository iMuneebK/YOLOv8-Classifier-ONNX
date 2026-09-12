import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# YOLOv8 configuration
YOLO_MODEL_CLASSIFICATION = "yolov8n-cls.pt"
YOLO_MODEL_DETECTION = "yolov8n.pt"

# Training parameters
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 640
LEARNING_RATE = 0.01

# UI settings
APP_TITLE = "YOLOv8 Image Classification & Detection"
