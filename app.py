import streamlit as st
import cv2
import numpy as np
from PIL import Image
from inference import YOLOInference
from utils.config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.sidebar.title("Configuration")

task = st.sidebar.selectbox("Task", ["Classification", "Detection"])
model_path = st.sidebar.text_input("Model Path", "yolov8n.pt" if task == "Detection" else "yolov8n-cls.pt")

@st.cache_resource
def load_model(path, task_type):
    return YOLOInference(path, task=task_type.lower())

try:
    inferencer = load_model(model_path, task)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting..." if task == "Detection" else "Classifying...")
    
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Run prediction
    results = inferencer.predict_image(img_array)
    
    st.subheader("Results")
    if task == "Detection":
        res_img = results.plot()
        st.image(res_img, caption="Detected Image", use_column_width=True)
    else:
        probs = results.probs
        st.write(f"**Top Class:** {results.names[probs.top1]}")
        st.write(f"**Confidence:** {probs.top1conf.item():.4f}")
