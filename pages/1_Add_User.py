import os.path

import streamlit as st
from PIL import Image

from verification_system import VerificationSystem

st.set_page_config(page_title="Add User", page_icon="👁️")


def add_user_images(user_name: str, user_images: list[bytes]):
    user_directory_path = os.path.join(
        "data", "database", "authorized_users", user_name
    )
    if not os.path.exists(user_directory_path):
        os.makedirs(user_directory_path, exist_ok=True)

    for user_image in user_images:
        Image.open(user_image).save(os.path.join(user_directory_path, user_image.name))


def show_user_images(user_name: str):
    user_directory_path = os.path.join(
        "data", "database", "authorized_users", user_name
    )
    user_images_path = os.listdir(user_directory_path)

    st.write(f"# Images of {user_name}")

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


st.title("Add User")

face_verification_system = VerificationSystem(
    database_path=os.path.join("data", "database")
)

uploaded_name = st.text_input(label="Username")

if uploaded_name:
    user_directory_path = os.path.join(
        "data", "database", "authorized_users", uploaded_name
    )
    if os.path.exists(user_directory_path):
        st.toast("Welcome back 👋")
    else:
        st.toast("Create profile by uploading images 👋")

    uploaded_images = st.file_uploader(
        label="Choose an image...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="hidden",
    )

    if uploaded_images:
        add_user_images(user_name=uploaded_name, user_images=uploaded_images)
        show_user_images(user_name=uploaded_name)
