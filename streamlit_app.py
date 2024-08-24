import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO('best.pt')  # Ensure this path is correct

st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This project is a web application for detecting common tomato leaf diseases.The model was trained on a specific dataset including various classes of tomato leaf diseases. The model classes are as follows:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

# Disease information dictionary
disease_info = {
    "Bacterial Spot": {
        "cause": "Caused by the bacterium Xanthomonas campestris pv. vesicatoria.",
        "cure": "Apply copper-based bactericides. Ensure proper crop rotation and avoid overhead watering.",
        "prevention": "Plant resistant varieties, avoid working in the garden when plants are wet, and use drip irrigation to reduce leaf wetness.",
        "chemicals": "Copper-based fungicides can help manage the disease."
    },
    "Early Blight": {
        "cause": "Caused by the fungus Alternaria solani.",
        "cure": "Use fungicides such as chlorothalonil or copper sprays. Remove affected leaves and practice crop rotation.",
        "prevention": "Practice crop rotation, space plants properly, and remove and destroy infected plant debris.",
        "chemicals": "Chlorothalonil, Mancozeb, and Copper-based fungicides."
    },
    "Healthy": {
        "cure": "No treatment needed.",
        "info": "The plant is healthy with no signs of disease.",
        "prevention": "Continue regular care and monitoring to maintain plant health.",
        "chemicals": "None needed."
    },
    "Iron Deficiency": {
        "cause": "Caused by insufficient iron in the soil, often due to high soil pH.",
        "cure": "Apply iron chelates to the soil or as a foliar spray.",
        "prevention": "Maintain proper soil pH, avoid over-watering, and use balanced fertilizers.",
        "chemicals": "Iron chelates, foliar sprays containing iron."
    },
    "Late Blight": {
        "cause": "Caused by the oomycete Phytophthora infestans.",
        "cure": "Use fungicides like mancozeb or chlorothalonil. Remove and destroy infected plants.",
        "prevention": "Plant resistant varieties, space plants properly, and avoid overhead watering.",
        "chemicals": "Mancozeb, Chlorothalonil, Copper fungicides."
    },
    "Leaf Mold": {
        "cause": "Caused by the fungus Passalora fulva.",
        "cure": "Ensure good air circulation, reduce humidity, and apply fungicides if necessary.",
        "prevention": "Provide good ventilation in greenhouses, water plants at the base, and remove infected leaves.",
        "chemicals": "Fungicides such as copper sprays or chlorothalonil."
    },
    "Leaf Miner": {
        "cause": "Caused by larvae of various leaf-mining insects.",
        "cure": "Use insecticides like spinosad or neem oil. Remove affected leaves.",
        "prevention": "Use row covers to prevent adult insects from laying eggs, and remove infested leaves.",
        "chemicals": "Spinosad, Neem oil."
    },
    "Mosaic Virus": {
        "cause": "Caused by several viruses, including the Tobacco Mosaic Virus (TMV).",
        "cure": "There is no cure for mosaic virus. Remove and destroy infected plants to prevent spread.",
        "prevention": "Use resistant varieties, avoid tobacco products while handling plants, and control aphids.",
        "chemicals": "No chemical treatment available."
    },
    "Septoria": {
        "cause": "Caused by the fungus Septoria lycopersici.",
        "cure": "Apply fungicides and remove infected leaves. Ensure good air circulation.",
        "prevention": "Avoid overhead watering, rotate crops, and remove plant debris.",
        "chemicals": "Chlorothalonil, Copper fungicides."
    },
    "Spider Mites": {
        "cause": "Caused by tiny spider-like pests, often due to dry, dusty conditions.",
        "cure": "Use miticides or insecticidal soaps. Maintain humidity to discourage mite infestations.",
        "prevention": "Maintain proper humidity, regularly mist plants, and introduce natural predators.",
        "chemicals": "Miticides, Insecticidal soaps."
    },
    "Yellow Leaf Curl Virus": {
        "cause": "Caused by the Tomato yellow leaf curl virus (TYLCV) transmitted by whiteflies.",
        "cure": "There is no cure. Remove and destroy infected plants. Control whiteflies to prevent spread.",
        "prevention": "Use reflective mulches to deter whiteflies, and use yellow sticky traps.",
        "chemicals": "Insecticides targeting whiteflies."
    }
}

def display_disease_info(detected_class):
    if detected_class in disease_info:
        st.sidebar.subheader(f"Disease Detected: {detected_class}")
        st.sidebar.markdown(f"**Cause:** {disease_info[detected_class]['cause']}")
        st.sidebar.markdown(f"**Cure:** {disease_info[detected_class]['cure']}")
        st.sidebar.markdown(f"**Prevention:** {disease_info[detected_class]['prevention']}")
        st.sidebar.markdown(f"**Chemicals:** {disease_info[detected_class]['chemicals']}")
    else:
        st.sidebar.subheader(f"Disease Detected: {detected_class}")
        st.sidebar.markdown("No information available for this disease.")

# Upload file
uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Predict classes
    results = model.predict(image)
    
    # Convert results to a format suitable for Streamlit display
    for result in results:
        im_array = result.plot()  # Plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        st.image(im, caption='Model Prediction')  # Show image with prediction
        
        # Extract and display disease information
        if len(result.boxes) > 0:
            detected_class_index = int(result.boxes[0].cls[0])  # Extract the index of the predicted class
            detected_class = result.names.get(detected_class_index, "Unknown")  # Safely get the class name
            display_disease_info(detected_class)
        else:
            st.subheader("No disease detected.")
        
        # Optionally, you can save the image as well
        im.save('results.jpg')

    # Success message
    st.success("Image processed successfully!")

    # Download processed image option
    with open("results.jpg", "rb") as file:
        btn = st.download_button(
            label="Download Processed Image",
            data=file,
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )
