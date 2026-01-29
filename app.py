import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.keras")

model = load_model()

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

st.title("Constellation Identifier")
st.write("Upload an image of a constellation")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {class_names[class_index]}")
    st.write(f"Confidence: {confidence:.2%}")
