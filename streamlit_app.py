
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

def main() :
    add_bg_from_url()
    menu = ["For Potatoes","For Tomatoes","Creators"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice=="For Potatoes":
        st.title(':red[Potato Leaf Disease Prediction]')
        file_uploaded = st.file_uploader('Choose an image...', type = 'jpg')
        if file_uploaded is not None :
            image = Image.open(file_uploaded)
            st.write("Uploaded Image.")
            figure = plt.figure()
            plt.imshow(image)
            plt.axis('off')
            st.pyplot(figure)
            result, confidence = predict_class_potato(image)
            st.write('Prediction : {}'.format(result))
            st.write('Confidence : {}%'.format(confidence))
    elif choice=="For Tomatoes":
        st.title(':red[Tomato Leaf Disease Prediction]')
        file_uploaded = st.file_uploader('Choose an image...', type = 'jpg')
        if file_uploaded is not None :
            image = Image.open(file_uploaded)
            st.write("Uploaded Image.")
            figure = plt.figure()
            plt.imshow(image)
            plt.axis('off')
            st.pyplot(figure)
            result, confidence = predict_class_tomato(image)
            st.write('Prediction : {}'.format(result))
            st.write('Confidence : {}%'.format(confidence))
    else:
        st.subheader(":blue[Project members]")
        st.subheader(":blue[Protim Aich]")
        st.subheader(":blue[Autonomy Roll Number - 12619001099]")
        st.subheader(":blue[Class Roll Number - 1951064]")
        st.subheader(":blue[Swapnil Chowdhury]")
        st.subheader(":blue[Autonomy Roll Number - 12619001172]")
        st.subheader(":blue[Class Roll Number - 1951065]")
        st.subheader(":blue[Anurag Nayak]")
        st.subheader(":blue[Autonomy Roll Number - 12619001035]")
        st.subheader(":blue[Class Roll Number - 1951011]")
        st.subheader(":blue[Shivam Shresth]")
        st.subheader(":blue[Autonomy Roll Number - 12619001145]")
        st.subheader(":blue[Class Roll Number - 1951001]")

def predict_class_potato(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'./models/CNN_Potato/final_cnn_model_potato_disease.h5', compile = False)

    shape = ((256,256,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)]) 
    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence
def predict_class_tomato(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'./models/CNN_Tomato/final_cnn_model_tomato_disease.h5', compile = False)

    shape = ((224,224,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)]) 
    test_image = image.resize((224,224))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-photo/abstract-luxury-gradient-blue-background-smooth-dark-blue-with-black-vignette-studio-banner_1258-63496.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}
a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
</div>
        """

st.markdown(footer, unsafe_allow_html = True)

if __name__ == '__main__' :
    main()
