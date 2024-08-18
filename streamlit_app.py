import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')  # Update with the correct path to your model

st.title("Tomato Leaf Disease Detection")
st.markdown("""
This project is a web application for detecting common tomato leaf diseases. It uses the YOLO (You Only Look Once) object detection model.
""")

# Webcam capture function
def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        st.warning("Could not access webcam.")
        return None

# Display webcam feed and capture button
if st.button("Capture Image from Webcam"):
    captured_image = capture_webcam()
    if captured_image is not None:
        st.image(captured_image, channels="BGR", caption="Captured Image")
        
        # Process the captured image with YOLO
        results = model(captured_image)
        
        # Display the processed image
        st.image(results[0].plot(), caption="Processed Image with Predictions", use_column_width=True)
