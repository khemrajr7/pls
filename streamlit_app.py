import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')  # Update with the correct path to your model

st.title("Tomato Leaf Disease Detection")
st.markdown("""
This project is a web application for detecting common tomato leaf diseases. It uses the YOLO (You Only Look Once) object detection model.
""")

# Upload file
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to array and run it through YOLO
    image_array = np.array(image)
    results = model(image_array)
    
    # Display the predictions
    st.image(results[0].plot(), caption="Processed Image with Predictions", use_column_width=True)
