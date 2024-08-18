import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import base64
import io

# Load the YOLO model
model = YOLO('best.pt')  # Ensure the correct path to your YOLO model is provided

st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This project is a web application for detecting common tomato leaf diseases. It uses the YOLO (You Only Look Once) object detection model. The model was trained on a specific dataset including various classes of tomato leaf diseases. The model classes are as follows:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

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
        const imageDataElement = document.getElementById('image-data');
        imageDataElement.value = dataUrl;
        imageDataElement.dispatchEvent(new Event('input', { bubbles: true }));
    });
</script>
"""

# Embed the HTML for webcam and capture button
st.markdown(webcam_html, unsafe_allow_html=True)

# Hidden text area to hold the image data
image_data = st.text_area("Captured Image Data URL", "", key="image-data")

# Process the image once captured
if image_data:
    image_data = image_data.split(",")[1]  # Remove the base64 header
    image_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_data))
    
    # Display the captured image
    st.image(image, caption="Captured Image")
    
    # Process the image using YOLO
    processed_image = process_image(np.array(image))
    st.image(processed_image, caption="Processed Image")

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

# Disease information dictionary
disease_info = {
    "Bacterial Spot": {
        "cure": "Apply copper-based bactericides. Ensure proper crop rotation and avoid overhead watering.",
        "info": "Bacterial spot causes dark, water-soaked spots on leaves and can lead to defoliation."
    },
    # Add more diseases and their corresponding information here...
}
