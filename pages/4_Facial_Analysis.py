import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace

st.set_page_config(page_title="Facial Analysis", page_icon="👁️")


def analyze_image(user_image: bytes):
    column_left, column_right = st.columns(2)

    with column_left:
        st.image(image=user_image, use_column_width="auto")

    try:
        image = Image.open(user_image)
        image_array = np.array(image)

        result = DeepFace.analyze(img_path=image_array)[0]

        with column_right:
            st.write("### Analysis Results:")
            st.write("Age:", result["age"])
            st.write("Gender:", result["dominant_gender"])
            st.write("Emotion:", result["dominant_emotion"])
            st.write("Race", result["race"])
    except Exception as e:
        with column_right:
            st.exception(e)


st.title("Facial Analysis")

uploaded_image = st.file_uploader(
    label="Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="hidden"
)

if uploaded_image is not None:
    analyze_image(user_image=uploaded_image)
