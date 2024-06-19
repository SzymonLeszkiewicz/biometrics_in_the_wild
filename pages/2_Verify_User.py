import os.path

import numpy as np
import streamlit as st
from PIL import Image

from verification_system import VerificationSystem

st.set_page_config(page_title="Verify User", page_icon="👁️")


def verify_user(user_name: str, user_image: bytes) -> bool:
    column_left, column_right = st.columns(2)

    with column_left:
        st.image(image=user_image, use_column_width="auto")

    try:
        image = Image.open(user_image)
        image_array = np.array(image)

        is_verified, _ = face_verification_system.verify_user(
            user_name=user_name, user_photo_path=image_array
        )

        with column_right:
            if is_verified:
                st.success("Verified")
            else:
                st.error("Not Verified")
        return is_verified
    except Exception as e:
        with column_right:
            st.exception(e)


def show_user_images(user_name: str):
    user_directory_path = os.path.join(
        "data", "database", "authorized_users", user_name
    )
    user_images_path = os.listdir(user_directory_path)

    st.write("# User Images")

    number_of_columns = 4
    number_of_images = len(user_images_path)
    number_of_rows = (number_of_images + number_of_columns - 1) // number_of_columns

    for row_index in range(number_of_rows):
        columns = st.columns(number_of_columns)
        for column_index in range(number_of_columns):
            index = row_index * number_of_columns + column_index
            if index < number_of_images:
                user_image_path = os.path.join(
                    user_directory_path, user_images_path[index]
                )
                columns[column_index].image(
                    image=user_image_path,
                    use_column_width="auto",
                )
            else:
                break


st.title("Verify User")

face_verification_system = VerificationSystem(
    database_path=os.path.join("data", "database")
)

uploaded_name = st.text_input(label="Username")

if uploaded_name:
    uploaded_image = st.file_uploader(
        label="Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="hidden",
    )

    if uploaded_image is not None:
        is_verified = verify_user(user_name=uploaded_name, user_image=uploaded_image)

        if is_verified:
            show_user_images(user_name=uploaded_name)
