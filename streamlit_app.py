import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('best.pt')

st.title("Tomato Leaf Disease Detection")

st.sidebar.header("About")
st.sidebar.markdown("""
This application helps in detecting common diseases in tomato leaves using the YOLO model. The model identifies various diseases and provides information on how to treat them.
""")

# Add custom CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .css-1aumxhk {
        padding: 1.5rem;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Upload file
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict classes
    results = model(image)

    # Convert results to a format suitable for Streamlit display
    im_array = results[0].plot()  # Plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
    st.image(im, caption='Model Prediction', use_column_width=True)

    # Extract and display disease information
    if len(results[0].boxes) > 0:
        detected_class_index = int(results[0].boxes[0].cls[0])  # Extract the index of the predicted class
        detected_class = results[0].names.get(detected_class_index, "Unknown")  # Safely get the class name
        
        # Get disease info
        if detected_class in disease_info:
            st.subheader(f"Disease Detected: {detected_class}")
            st.markdown(f"**Cure:** {disease_info[detected_class]['cure']}")
            st.markdown(f"**Additional Info:** {disease_info[detected_class]['info']}")
        else:
            st.subheader(f"Disease Detected: {detected_class}")
            st.markdown("No information available for this disease.")
    else:
        st.subheader("No disease detected.")

    # Optionally, you can save the image
    im.save('results.jpg')
