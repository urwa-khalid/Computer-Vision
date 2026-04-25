# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import algorithms as algo

st.set_page_config(page_title="Computer Vision Portfolio", layout="wide")

# Sidebar for Navigation
st.sidebar.title("CV Assignments")
mode = st.sidebar.selectbox("Select Project", 
    ["360° Rotation", "Quantization & IGS", "Height Measurement"])

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Convert upload to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_rgb)

    with col2:
        st.subheader("Processed Result")
        
        if mode == "360° Rotation":
            angle = st.slider("Select Angle", 0, 360, 0)
            result = algo.rotation(img_rgb, angle)
            st.image(result)

        elif mode == "Quantization & IGS":
            bits = st.select_slider("Select Bit Depth", options=[1, 2, 4, 8], value=4)
            levels = 2**bits
            q_mode = st.radio("Quantization Mode", ["Lower", "Middle", "Higher"])
            enable_igs = st.checkbox("Apply IGS (Improved Gray Scale)")
            
            if enable_igs:
                result = algo.igs_quantization(img_rgb, levels)
            else:
                result = algo.gray_quantization(img_rgb, levels, q_mode)
            st.image(result, clamp=True)
            
        elif mode == "Height Measurement":
            st.info("Height measurement requires camera calibration data.")
            st.write("Current implementation: Placeholder for calibration logic.")