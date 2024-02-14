import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# creating the title
st.title("Covid-19 Chest X-ray Classifier")

# creating a side bar 
st.sidebar.title("Created By:")
st.sidebar.subheader("P.S.S.Keerthana")
st.sidebar.subheader("P.Komal Sai Anurag")
st.sidebar.subheader("Udayagiri Varun")
st.sidebar.subheader("Sejal Singh")
st.sidebar.subheader(" ")

st.sidebar.image("https://post.healthline.com/wp-content/uploads/2020/08/chest-x-ray_thumb.jpg", width=None)

# creating an uploader to upload the Chest X-ray images
upload_file = st.file_uploader("Upload the Chest X-ray", type = ['jpg','png','jpeg'])

# creating a predict button
generate_pred = st.button("Predict")

model = tf.keras.models.load_model('covid_classifier.h5')


def import_n_pred(image_data,model):
    size = (128,128)
    image = cv2.resize(image_data,size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0) 
    pred = model.predict(image)
    return pred

if generate_pred:
    image = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), 1)
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image,model)
    labels = ['Healthy','Covid-19']
    st.title("The Chest X-ray is {}".format(labels[np.argmax(pred)]))
