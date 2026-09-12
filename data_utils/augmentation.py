import cv2
import albumentations as A
import numpy as np

def get_training_augmentation():
    """Returns albumentations composed transform for training."""
    return A.Compose([
        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Rotate(limit=30, p=0.5)
    ])

def apply_augmentation(image: np.ndarray) -> np.ndarray:
    """Apply training augmentation to a single image."""
    transform = get_training_augmentation()
    augmented = transform(image=image)
    return augmented['image']
