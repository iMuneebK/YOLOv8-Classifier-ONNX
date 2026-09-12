# YOLOv8 Image Classification & Object Detection Pipeline 🚀

![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)

A comprehensive, end-to-end computer vision pipeline using YOLOv8 for both **Image Classification** and **Object Detection**. This project demonstrates professional ML engineering practices including custom dataset preparation, data augmentation, model training, evaluation, inference across multiple media types, and deployment via Streamlit.

## 📌 Features

- **Transfer Learning with YOLOv8:** Fine-tune state-of-the-art YOLOv8 models on custom datasets.
- **Dual Support:** Pipelines for both Classification (`train_classifier.py`) and Detection (`train_detector.py`).
- **Data Engineering:** Automated dataset splitting and data augmentation utilities (Albumentations).
- **Evaluation Metrics:** Confusion matrices, precision-recall visualization, and training history plotting.
- **Versatile Inference:** Run inference on images, video streams, or live webcam.
- **Streamlit Web App:** Interactive UI for uploading images and viewing model predictions in real-time.
- **Deployment-Ready:** Scripts to export trained PyTorch models to ONNX format.

## 🏗 YOLOv8 Architecture Overview

YOLOv8 by Ultralytics introduces an anchor-free detection paradigm, reducing the number of box predictions and speeding up non-maximum suppression (NMS). It utilizes a modified CSPDarknet53 backbone and a specialized Decoupled Head for object detection, separating classification and regression tasks. For the classification variant, it leverages the same robust backbone tailored for categorical prediction.

### Transfer Learning Process
1. **Feature Extraction:** Pre-trained weights (trained on ImageNet/COCO) are loaded.
2. **Fine-Tuning:** The model head is replaced or adapted to match the number of classes in the custom dataset.
3. **Optimization:** SGD or AdamW optimizer fine-tunes the network with learning rate schedulers to maximize metric performance (mAP for detection, Accuracy for classification).

## 🚀 Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/yourusername/ai-image-classifier-yolov8.git
cd ai-image-classifier-yolov8
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Dataset Preparation
Organize classification data into `data/raw/` in class folders.
```bash
python data_utils/prepare_dataset.py
```

### 3. Training
**For Classification:**
```bash
python train_classifier.py
```
**For Detection:**
```bash
python train_detector.py
```

### 4. Run Streamlit UI
```bash
streamlit run app.py
```

## 📊 Training Results
During training, YOLOv8 generates extensive logs and plots in the `runs/` directory. Our evaluation utilities (`evaluation/visualize.py`) can further process these to show:
- Precision-Recall (PR) Curves
- Confusion Matrices highlighting per-class accuracy
- Train/Validation Loss Curves over Epochs

## 📸 Demo
Launch the `app.py` UI to interactively test the model. Simply select your model task, provide the weights file, and upload an image!

## 💾 Model Export
To deploy the model using ONNX Runtime for faster inference:
```bash
python export/onnx_export.py models/classifier_run/weights/best.pt
```
