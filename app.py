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

# Function to resize the image and pad to keep aspect ratio
def resize_and_pad_image(image, target_size=(640, 640)):
    original_size = image.size
    image = image.convert("RGB")

    # Resize the image while keeping the aspect ratio
    image.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    new_image.paste(image, ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2))
    
    return new_image

# Function to perform detection and plot results
def detect_and_plot(image, model):
    # Convert the image to NumPy array
    image_np = np.array(image)

    # Perform detection
    results = model.predict(image_np)[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    
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

# Custom HTML and CSS styling
st.markdown("""
    <style>
        .big-title {
            font-size: 48px; /* Double the default size */
            color: #FF0800; /* Red color */
            text-align: center;
        }
        .sub-title {
            font-size: 24px; /* Half the size of the page title */
            color: #FF0800; /* Red color */
        }
        .output-title {
            font-size: 24px; /* Half the size of the page title */
            color: #FF0800; /* Red color */
        }
    </style>
    <h1 class="big-title">Brain Tumor Detection</h1>
""", unsafe_allow_html=True)

# File uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and resize the image using PIL
    image = Image.open(uploaded_image)
    image_resized = resize_and_pad_image(image, (640, 640))  # Resize and pad image to 640x640
    
    # Display the resized image with a heading
    st.markdown('<h2 class="sub-title">Input Image</h2>', unsafe_allow_html=True)
    st.image(image_resized)

    # Load the YOLO model
    model_path = 'yolov8_model.pt'  # Update this path to your model
    model = load_model(model_path)
    
    if model is not None:
        # Perform detection and get the result plot
        result_plot = detect_and_plot(image_resized, model)
        
        # Display the result plot with a heading above it
        st.markdown('<h2 class="output-title">Detection Results</h2>', unsafe_allow_html=True)
        st.image(result_plot)
