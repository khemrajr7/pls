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
    "Early Blight": {
        "cure": "Use fungicides such as chlorothalonil or copper sprays. Remove affected leaves and practice crop rotation.",
        "info": "Early blight appears as dark, concentric rings on older leaves, usually starting at the bottom of the plant."
    },
    "Healthy": {
        "cure": "No treatment needed.",
        "info": "The plant is healthy with no signs of disease."
    },
    "Iron Deficiency": {
        "cure": "Apply iron chelates to the soil or as a foliar spray.",
        "info": "Iron deficiency leads to yellowing between the veins of young leaves, while veins remain green."
    },
    "Late Blight": {
        "cure": "Use fungicides like mancozeb or chlorothalonil. Remove and destroy infected plants.",
        "info": "Late blight causes dark, water-soaked lesions on leaves and stems, leading to plant collapse."
    },
    "Leaf Mold": {
        "cure": "Ensure good air circulation, reduce humidity, and apply fungicides if necessary.",
        "info": "Leaf mold appears as yellow spots on the upper leaf surface and mold growth on the underside."
    },
    "Leaf Miner": {
        "cure": "Use insecticides like spinosad or neem oil. Remove affected leaves.",
        "info": "Leaf miners create winding, white trails on leaves as they feed inside the leaf tissue."
    },
    "Mosaic Virus": {
        "cure": "There is no cure for mosaic virus. Remove and destroy infected plants to prevent spread.",
        "info": "Mosaic virus causes mottled, yellow, or white patterns on leaves and stunted growth."
    },
    "Septoria": {
        "cure": "Apply fungicides and remove infected leaves. Ensure good air circulation.",
        "info": "Septoria leaf spot presents as small, circular spots with dark borders on lower leaves."
    },
    "Spider Mites": {
        "cure": "Use miticides or insecticidal soaps. Maintain humidity to discourage mite infestations.",
        "info": "Spider mites cause stippling and yellowing of leaves, often with webbing on the undersides."
    },
    "Yellow Leaf Curl Virus": {
        "cure": "There is no cure. Remove and destroy infected plants. Control whiteflies to prevent spread.",
        "info": "Yellow leaf curl virus causes yellowing and upward curling of leaves, stunted growth, and reduced yield."
    }
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
def capture_from_webcam():
    cap = cv2.VideoCapture(0)  # Open the webcam
    stframe = st.empty()  # Placeholder for the video frames

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
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
    capture_from_webcam()
