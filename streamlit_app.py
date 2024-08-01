import streamlit as st
from PIL import Image
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
model = YOLO('best.pt')

st.title("Tomato Leaf Disease Detection")
st.markdown("""
*This project is a web application for detecting common tomato leaf diseases. It uses the YOLO (You Only Look Once) object detection model. The model was trained on a specific dataset including various classes of tomato leaf diseases. The model classes are as follows:
Bacterial Spot, Early Blight, Healthy, Iron Deficiency, Late Blight, Leaf Mold, Leaf Miner, Mosaic Virus, Septoria, Spider Mites, Yellow Leaf Curl Virus.*
""")

uploaded_file = st.file_uploader("Choose a tomato leaf image", type=["jpg", "png"])

# Add an author section
st.sidebar.markdown("## Author")
st.sidebar.markdown("Name: **Arix ALIMAGNIDOKPO**")
st.sidebar.markdown("GitHub: https://github.com/Arix-ALIMAGNIDOKPO")
st.sidebar.markdown("LinkedIn: www.linkedin.com/in/arix-alimagnidokpo-27865a276")
st.sidebar.markdown("## For more information, please check the GitHub repository link below:")
st.sidebar.markdown("Repository: https://github.com/Arix-ALIMAGNIDOKPO/Tomato-Leaf-Disease-Dection-using-Yolov8")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Predict classes
    results = model(image)
    # View results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image
    st.image('results.jpg', caption='Model Prediction')
