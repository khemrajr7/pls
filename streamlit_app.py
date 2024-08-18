import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import base64
import io
import json

# Load the YOLO model
model = YOLO('best.pt')  # Make sure to specify the correct path to your trained YOLO model

st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This project is a web application for detecting common tomato leaf diseases. It uses the YOLO (You Only Look Once) object detection model. The model was trained on a specific dataset including various classes of tomato leaf diseases. The model classes are as follows:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

# Function to process the image and make predictions
def process_image(image):
    results = model(image)
    for result in results:
        im_array = result.plot()  # Plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        
        if len(result.boxes) > 0:
            detected_class_index = int(result.boxes[0].cls[0])  # Extract the index of the predicted class
            detected_class = result.names.get(detected_class_index, "Unknown")  # Safely get the class name
            
            if detected_class in disease_info:
                cure = disease_info[detected_class]["cure"]
                info = disease_info[detected_class]["info"]
                st.subheader(f"Disease Detected: {detected_class}")
                st.markdown(f"**Cure:** {cure}")
                st.markdown(f"**Additional Info:** {info}")
            else:
                st.subheader(f"Disease Detected: {detected_class}")
                st.markdown("No information available for this disease.")
        else:
            st.subheader("No disease detected.")
        return im

# Define a dictionary with disease information
disease_info = {
    "Bacterial Spot": {
        "cure": "Apply copper-based bactericides. Ensure proper crop rotation and avoid overhead watering.",
        "info": "Bacterial spot causes dark, water-soaked spots on leaves and can lead to defoliation."
    },
    # Add other disease information here...
}

# HTML and JS code to access the webcam and take a photo
webcam_html = """
<div>
    <video id="video" width="100%" height="100%" autoplay></video>
    <button id="snap">Capture Image</button>
    <canvas id="canvas" style="display:none;"></canvas>
</div>
<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const snapButton = document.getElementById('snap');
    const context = canvas.getContext('2d');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
        });
    snapButton.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/png');
        const jsonData = JSON.stringify({ image: dataUrl });
        fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: jsonData
        }).then(response => response.json())
          .then(data => {
              console.log(data);
              window.location.reload();
          });
    });
</script>
"""

st.markdown(webcam_html, unsafe_allow_html=True)

# Check if there's image data in the session state
if "image_data" in st.session_state:
    image_data = st.session_state["image_data"]
    image = Image.open(io.BytesIO(image_data))
    
    # Display the image
    st.image(image, caption="Captured Image")

    # Process the image
    processed_image = process_image(np.array(image))
    st.image(processed_image, caption="Processed Image")

# Handle the image data sent via POST
if st.experimental_get_query_params().get('image'):
    image_data_url = st.experimental_get_query_params()['image'][0]
    image_data = base64.b64decode(image_data_url.split(",")[1])  # Remove the base64 header
    st.session_state["image_data"] = image_data
