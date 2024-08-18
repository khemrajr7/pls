import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

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

# Webcam capture function
def capture_from_webcam(camera_index=0):
    cap = cv2.VideoCapture(camera_index)  # Use the working camera index found in the test
    
    if not cap.isOpened():
        st.error("Could not access the webcam. Please check the camera index or connection.")
        return
    
    stframe = st.empty()  # Placeholder for the video frames

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            st.error("Failed to capture image from webcam. Please try again.")
            break
        
        # Convert the frame to RGB (OpenCV captures in BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        im = process_image(frame_rgb)
        
        # Display the frame
        stframe.image(im, channels="RGB")
        
        if st.button("Stop"):
            break

    cap.release()  # Release the webcam when done

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose the mode", ["Upload Image", "Use Webcam"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        im = process_image(np.array(image))
        st.image(im, caption='Model Prediction')
elif mode == "Use Webcam":
    st.text("Webcam capture mode")
    capture_from_webcam(camera_index=0)  # Using the correct index
