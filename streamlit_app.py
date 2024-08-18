import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import base64
import io

# Load the YOLO model
model = YOLO('best.pt')  # Make sure to specify the correct path to your trained YOLO model

st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This project is a web application for detecting common tomato leaf diseases. It uses the YOLO (You Only Look Once) object detection model. The model was trained on a specific dataset including various classes of tomato leaf diseases. The model classes are as follows:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

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
<div style="display: flex; justify-content: space-between;">
    <div>
        <video id="video" width="300" height="300" autoplay></video>
        <button id="snap" style="margin-top: 10px;">Capture Image</button>
    </div>
    <div>
        <canvas id="canvas" width="300" height="300" style="border:1px solid #d3d3d3;"></canvas>
    </div>
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
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/png');
        document.getElementById('image-data').value = dataUrl;
        document.getElementById('submit-image').click();  // Automatically trigger the submit button
    });
</script>
"""

st.markdown(webcam_html, unsafe_allow_html=True)

# Hidden input and submit button for captured image data
st.write('<input type="hidden" id="image-data" name="image-data">', unsafe_allow_html=True)
submit_button = st.button("Process Captured Image", key="submit-image")

# Handling the captured image data
if submit_button:
    image_data_url = st.session_state.get('image_data')
    if image_data_url:
        image_data = image_data_url.split(",")[1]  # Remove the base64 header
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Display the image
        st.image(image, caption="Captured Image")

        # Process the image
        processed_image = process_image(np.array(image))
        st.image(processed_image, caption="Processed Image")
