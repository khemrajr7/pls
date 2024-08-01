import streamlit as st
from PIL import Image
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
model = YOLO('best.pt')

st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This project is a web application for detecting common tomato leaf diseases using the YOLO (You Only Look Once) object detection model. The model has been trained on a specific dataset containing different classes of tomato leaf diseases. The model classes include:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])



if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Predict classes
    results = model(image)
    
    # View results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('results.jpg')  # save image
    
    st.image('results.jpg', caption='Model Prediction')
    
    detected_class = results[0].names[0]  # assuming the first result is the detected class

    disease_info = {
        "Bacterial Spot": "Remedy: Use copper-based fungicides and avoid overhead watering.",
        "Early Blight": "Remedy: Remove affected leaves and apply fungicides.",
        "Healthy": "Your plant is healthy!",
        "Iron Deficiency": "Remedy: Apply iron chelate to the soil.",
        "Late Blight": "Remedy: Remove infected plants and use fungicides.",
        "Leaf Mold": "Remedy: Improve air circulation and apply fungicides.",
        "Leaf Miner": "Remedy: Use neem oil or insecticidal soap.",
        "Mosaic Virus": "Remedy: Remove infected plants and control aphids.",
        "Septoria": "Remedy: Remove affected leaves and use fungicides.",
        "Spider Mites": "Remedy: Use insecticidal soap or horticultural oil.",
        "Yellow Leaf Curl Virus": "Remedy: Remove infected plants and control whiteflies."
    }

    st.markdown(f"### Detected Disease: {detected_class}")
    st.markdown(f"### Remedy: {disease_info.get(detected_class, 'No remedy available.')}")
