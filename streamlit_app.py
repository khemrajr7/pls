import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io
import numpy as np

# Load the YOLO model
model = YOLO('best.pt')  # Make sure to specify the correct path to your trained YOLO model

st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This project is a web application for detecting common tomato leaf diseases. It uses the YOLO (You Only Look Once) object detection model. The model was trained on a specific dataset including various classes of tomato leaf diseases. The model classes are as follows:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")



# Upload file
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Predict classes
    results = model(image)
    
    # Convert results to a format suitable for Streamlit display
    for result in results:
        im_array = result.plot()  # Plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        st.image(im, caption='Model Prediction')  # Show image with prediction
        
        # Optionally, you can save the image as well
        im.save('results.jpg')
