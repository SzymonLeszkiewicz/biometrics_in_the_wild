import os.path

import numpy as np
import streamlit as st

from verification_system import VerificationSystem

st.set_page_config(page_title="Verify Multiple User", page_icon="👁️")


@st.cache_data
def verify_multiple_users(incoming_users_path: str):
    try:
        return face_verification_system.verify_multiple_users(
            incoming_users_path=incoming_users_path
        )
    except Exception as e:
        st.exception(e)


face_verification_system = VerificationSystem(
    database_path=os.path.join("data", "database")
)

st.title("Verify Multiple Users")

use_mode = st.toggle(label="Use Authorized/Unauthorized Users Mode")

if not use_mode:
    uploaded_folder_path = st.text_input(
        "Enter path to the folder containing users profiles:"
    )

    if os.path.isdir(uploaded_folder_path):
        df_multiple_users = verify_multiple_users(uploaded_folder_path)
        access_granted_rate = face_verification_system.calculate_access_granted_rate(
            df_multiple_users
        )

        st.write("Access Granted Rate:", np.round(access_granted_rate, 3))
        st.dataframe(df_multiple_users)

else:
    uploaded_authorized_users_folder_path = st.text_input(
        "Enter path to the folder containing authorized users profiles:"
    )
    uploaded_unauthorized_users_folder_path = st.text_input(
        "Enter path to the folder containing unauthorized users profiles:"
    )

    if os.path.isdir(uploaded_authorized_users_folder_path) and os.path.isdir(
        uploaded_unauthorized_users_folder_path
    ):
        df_authorized_users = verify_multiple_users(
            uploaded_authorized_users_folder_path
        )
        df_unauthorized_users = verify_multiple_users(
            uploaded_unauthorized_users_folder_path
        )
        false_acceptance_rate, false_rejection_rate = (
            face_verification_system.calculate_far_frr(
                df_authorized_users, df_unauthorized_users
            )
        )

        true_negative, false_positive, false_negative, true_positive = (
            face_verification_system.calculate_ROC_curve(
                df_authorized_users,
                df_unauthorized_users,
            )
        )
        accuracy = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )
        access_granted_rate = face_verification_system.calculate_access_granted_rate(
            df_authorized_users
        )

        st.write("## Results")

        column_left, column_right = st.columns(2)

        with column_left:
            st.write("False Acceptance Rate: ", np.round(false_acceptance_rate, 3))
            st.write("False Rejection Rate: ", np.round(false_rejection_rate, 3))
            st.write("Accuracy: ", np.round(accuracy, 3))
            st.write("Access Granted Rate: ", np.round(access_granted_rate, 3))

        with column_right:
            st.write("True Negative: ", true_negative)
            st.write("False Positive: ", false_positive)
            st.write("False Negative: ", false_negative)
            st.write("True Positive: ", true_positive)

        st.write("### Authorized Users")
        st.dataframe(df_authorized_users)

        st.write("### Unauthorized Users")
        st.dataframe(df_unauthorized_users)
