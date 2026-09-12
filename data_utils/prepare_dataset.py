import os
import shutil
from sklearn.model_selection import train_test_split

def create_dataset_splits(data_dir: str, output_dir: str, test_size: float = 0.2):
    """Split dataset into train and validation sets."""
    os.makedirs(output_dir, exist_ok=True)
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        images = os.listdir(cls_dir)
        
        train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=42)
        
        # Create train/val directories
        os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', cls), exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(output_dir, 'train', cls, img))
            
        for img in val_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(output_dir, 'val', cls, img))
            
    print(f"Dataset split completed. Output saved to {output_dir}")

if __name__ == "__main__":
    create_dataset_splits("data/raw", "data/processed")
