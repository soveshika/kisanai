import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict
from treatment_advice import get_advice
from gradcam import save_gradcam

st.set_page_config(
    page_title="KisanAI - Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 KisanAI")
st.subheader("Crop Disease Detection for Indian Farmers")
st.write("फसल रोग पहचान प्रणाली | Upload a leaf photo to get instant diagnosis")

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    temp_path = "temp_leaf.jpg"
    image.save(temp_path)

    with st.spinner("Analysing your crop..."):
        disease, confidence = predict(temp_path)
        advice = get_advice(disease)
        gradcam_path = save_gradcam(temp_path, "gradcam_output.jpg")

    st.success("Analysis Complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Disease Detected", advice['english_name'])
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    st.markdown(f"### रोग: {advice['hindi_name']}")

    st.image("gradcam_output.jpg", caption="Grad-CAM: Where the model looked", use_column_width=True)

    st.markdown("---")
    st.markdown("### 💊 Chemical Treatment")
    st.info(advice['chemical'])

    st.markdown("### 🌱 Organic Treatment")
    st.success(advice['organic'])

    st.markdown("### 🛡️ Prevention")
    st.warning(advice['prevention'])

    st.markdown("### 🇮🇳 Hindi Advice | हिंदी सलाह")
    st.markdown(f"**{advice['hindi_advice']}**")

    os.remove(temp_path)