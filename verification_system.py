import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm.autonotebook import tqdm


class VerificationSystem:
    def __init__(
        self,
        database_path: str,
        destination: str = "authorized_users",
        acceptance_threshold: float = 0.5,
        model_name: str = "Facenet512",
        allow_brute_force: bool = False,
        example_initalizing_image_path: str = None,
    ):
        self.database_path = database_path
        self.destination = destination
        self.acceptance_threshold = acceptance_threshold
        self.model_name = model_name
        self.allow_brute_force = allow_brute_force

        self.example_initializing_image_path = (
            example_initalizing_image_path or self.get_incoming_authorized_user_path()
        )

        self.initialize_database()

    def initialize_database(self) -> None:
        DeepFace.find(
            img_path=self.example_initializing_image_path,
            db_path=os.path.join(self.database_path, self.destination),
            threshold=self.acceptance_threshold,
            enforce_detection=False,
            model_name=self.model_name,
        )

    def verify_user(
        self, user_name: str, user_photo_path: str | np.ndarray
    ) -> Tuple[bool, float]:
        faces_found = DeepFace.find(
            img_path=user_photo_path,
            db_path=os.path.join(self.database_path, self.destination),
            threshold=self.acceptance_threshold,
            enforce_detection=False,
            silent=True,
            model_name=self.model_name,
        )

        # no face detected or above acceptance threshold
        if faces_found[0].empty:
            return False, np.inf

        # TODO: find a way to make it path independent
        # assumption that only one face is in the image
        predicted_user_name = faces_found[0]["identity"].apply(
            lambda x: Path(x).parts[3]
        )[
            0
        ]  # get the distance closest match

        is_access_granted = user_name == predicted_user_name

        if self.allow_brute_force:
            system_user_names = os.listdir(
                os.path.join(self.database_path, self.destination)
            )

            if user_name in system_user_names:
                system_user_names.remove(user_name)

            is_access_granted = predicted_user_name in system_user_names

        return is_access_granted, faces_found[0]["distance"]

    def verify_multiple_users(self, incoming_users_path: str) -> pd.DataFrame:
        df_users = pd.DataFrame(
            columns=[
                "image_path",
                "is_access_granted",
                "distance",
            ]
        )

        for user_name in tqdm(
            iterable=os.listdir(incoming_users_path), desc="Processing Users"
        ):
            for user_photo in tqdm(
                iterable=os.listdir(os.path.join(incoming_users_path, user_name)),
                desc="Processing Images",
                leave=False,
            ):
                is_access_granted, distance = self.verify_user(
                    user_name=user_name,
                    user_photo_path=os.path.join(
                        incoming_users_path, user_name, user_photo
                    ),
                )

                df_user = pd.DataFrame(
                    {
                        "image_path": os.path.join(
                            incoming_users_path, user_name, user_photo
                        ),
                        "is_access_granted": is_access_granted,
                        "distance": distance,
                    },
                    index=[0],
                )

                df_users = pd.concat([df_users, df_user], ignore_index=True)

        return df_users

    @staticmethod
    def calculate_access_granted_rate(
        df_users: pd.DataFrame,
    ) -> float:
        return df_users["is_access_granted"].sum() / len(df_users)

    def calculate_ROC_curve(
        self,
        df_users_authorized: pd.DataFrame,
        df_users_unauthorized: pd.DataFrame,
        roc_curve_path: str = None,
    ) -> Tuple[int, int, int, int]:
        """
        Function to draw ROC curve based on DFs with authorized and unauthorized users, based on changing threshold
        of distance.
        :param df_users_authorized: DF with users in database
        :param df_users_unauthorized: DF with users that are not authorized in database
        :param roc_curve_path: path to save ROC curve plot
        :return: Tuple of TN, FP, FN, TP
        """
        df_concatenated = pd.concat(
            [df_users_authorized, df_users_unauthorized], axis=0
        )
        true_labels = [True] * len(df_users_authorized) + [False] * len(
            df_users_unauthorized
        )
        predicted_labels = df_concatenated["is_access_granted"].to_list()
        distances = df_concatenated["distance"]

        non_inf_values = distances.replace([np.inf, -np.inf], np.nan).dropna().unique()
        non_inf_values_sorted = np.sort(non_inf_values)[::-1]
        second_max_value = (
            non_inf_values_sorted[1]
            if len(non_inf_values_sorted) > 1
            and len(non_inf_values) != len(distances.unique())
            else non_inf_values_sorted[0]
        )  # get valid distances from system
        distances.replace(
            [np.inf], second_max_value, inplace=True
        )  # replace inf distance with max available distance, which is not np.inf -> system errored finding match
        probabilities = (
            second_max_value - distances
        )  # analyze distances as probabilities of accepting by system

        acceptance_threshold_scaled = (
            self.acceptance_threshold - probabilities.min()
        ) / (probabilities.max() - probabilities.min())
        scale_factor = (
            0.5 / acceptance_threshold_scaled
        )  # make sure that threshold in in the middle of probabilities
        probabilities_rescaled = (probabilities - probabilities.min()) * scale_factor

        if roc_curve_path is not None:
            roc_curve_directory = os.path.dirname(roc_curve_path)
            if not os.path.exists(roc_curve_directory):
                os.makedirs(roc_curve_directory, exist_ok=True)

            fpr, tpr, thresholds = roc_curve(true_labels, probabilities_rescaled)
            plt.figure(figsize=(16, 9), dpi=200)
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.tight_layout()
            plt.savefig(roc_curve_path, dpi=200)
            plt.close()

        tn, fp, fn, tp = confusion_matrix(
            y_true=true_labels, y_pred=predicted_labels
        ).ravel()
        return tn, fp, fn, tp

    @staticmethod
    def calculate_far_frr(
        df_users_authorized: pd.DataFrame, df_users_unauthorized: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Function to calculate False Acceptance Rate, False Rejection Rate

        :param df_users_authorized: DF with users in database
        :param df_users_unauthorized: DF with users that are not authorized in database
        :return: False Acceptance Rate, False Rejection Rate
        """
        df_concatenated = pd.concat(
            [df_users_authorized, df_users_unauthorized], axis=0
        )
        true_labels = [True] * len(df_users_authorized) + [False] * len(
            df_users_unauthorized
        )
        predicted_labels = df_concatenated["is_access_granted"].to_list()
        tn, fp, fn, tp = confusion_matrix(
            y_true=true_labels, y_pred=predicted_labels
        ).ravel()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        frr = 1 - tpr
        far = fpr
        return far, frr

    def get_incoming_authorized_user_path(self) -> str:
        return os.path.join(
            self.database_path, "incoming_users", "authorized_users", "1", "000023.jpg"
        )

    def get_incoming_unauthorized_user_path(self):
        return os.path.join(
            self.database_path,
            "incoming_users",
            "unauthorized_users",
            "101",
            "020633.jpg",
        )

    def get_problematic_incoming_authorized_user_path(self):
        return os.path.join(
            self.database_path, "incoming_users", "authorized_users", "22", "001677.jpg"
        )
