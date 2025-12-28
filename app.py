import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO

st.set_page_config(
    page_title="Cat vs Dog Classifier 🐱🐶",
    page_icon="🐶",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "cat_and_dog_classifier.h5")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    if confidence > 0.5:
        return "🐶 Dog", confidence
    else:
        return "🐱 Cat", 1 - confidence

st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image or paste an image URL")

option = st.radio(
    "Choose input method:",
    ["Upload Image", "Image URL"]
)

with st.spinner("Loading model..."):
    model = load_model()

st.success("Model loaded ✅")

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, width=350)

        label, confidence = predict_image(img)

        st.subheader(f"Prediction: {label}")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence*100:.2f}%**")

if option == "Image URL":
    image_url = st.text_input("Paste image URL here")

    if image_url:
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            st.image(img, width=350)

            label, confidence = predict_image(img)

            st.subheader(f"Prediction: {label}")
            st.progress(confidence)
            st.write(f"Confidence: **{confidence*100:.2f}%**")

        except:
            st.error("Invalid image URL")

st.markdown("---")
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        Built with ❤️ using Streamlit & TensorFlow <br>
        Built by <strong>Mujtaba Ahmed 💻</strong> • 
        <a href='https://github.com/mujtabaansari404' target='_blank'>GitHub 🚀</a>
    </div>
    """,
    unsafe_allow_html=True
)

