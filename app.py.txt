import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("cnn_model.h5")

# EXACT class order from training
CLASS_NAMES = [
    "cassiopeia",
    "crux",
    "cygnus",
    "gemini",
    "leo",
    "orion",
    "scorpius",
    "ursa_major"
]

st.set_page_config(page_title="Constellation Classifier", page_icon="🌌")

st.title("🌌 Star Constellation Identifier")
st.write("Upload an image of a star constellation and let the model identify it.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (must match training)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    st.success(f"🌟 Prediction: **{CLASS_NAMES[class_index].title()}**")
    st.write(f"Confidence: **{confidence:.2%}**")
