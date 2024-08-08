import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import io

# Define the classes
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Function to perform detection and plot results
def detect_and_plot(image, model):
    results = model.predict(image)[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    
    for detection in results.boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf[0].cpu().numpy()
        cls = detection.cls[0].cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f"{classes[int(cls)]} {conf:.2f}", color='white', fontsize=12, backgroundcolor='red')
        
    plt.axis('off')
    
    # Save the plot to a BytesIO object to display in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Streamlit app setup
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF0800;'>Brain Tumor Detection</h1>", unsafe_allow_html=True)

st.subheader("Upload Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image using PIL
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert PIL image to a format suitable for YOLO model
    image_np = np.array(image)
    
    # Load the YOLO model
    model_path = 'yolov8_model.pt'  # Update this path to your model
    model = load_model(model_path)
    
    if model is not None:
        # Perform detection and get the result plot
        result_plot = detect_and_plot(image_np, model)
        
        # Display the result plot in Streamlit
        st.image(result_plot, caption='Detection Results')
