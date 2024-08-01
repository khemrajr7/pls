import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load models
models = {
    'Inception': tf.keras.models.load_model('inc.h5'),
    'CNN': tf.keras.models.load_model('cnn.h5')
}

# Define classes
classes = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
           'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite']


def load_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return classes[predicted_class[0]]


st.title('Tomato Leaf Disease Prediction')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_image(uploaded_file)
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Model selection
    option = st.selectbox('Which model would you like to use?', ('Inception', 'CNN', 'Sequential'))
    model = models[option]

    # Perform prediction
    if st.button('Predict'):
        prediction = predict(model, img)
        st.write(f'Prediction: {prediction}')

        # Display remedies based on the predicted disease
        remedies = {
            'Tomato___Bacterial_spot': "Use copper fungicides to manage the bacterial spot.",
            'Tomato___Early_blight': "Apply fungicides and practice crop rotation.",
            'Tomato___healthy': "No disease detected. Your plant is healthy.",
            'Tomato___Late_blight': "Remove and destroy affected plants, and apply fungicides.",
            'Tomato___Leaf_Mold': "Ensure good air circulation and apply fungicides.",
            'Tomato___Septoria_leaf_spot': "Remove infected leaves and apply fungicides.",
            'Tomato___Spider_mites Two-spotted_spider_mite': "Use insecticidal soaps or oils."
        }

        st.write("Remedies:")
        st.write(remedies[prediction])
else:
    st.write("Please upload an image to classify.")
