"Code has its source in repository: https://github.com/Praveenanand333/Snapchat-Filters/tree/main"

import itertools
import warnings

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles


def detectFacialLandmarks(image, face_mesh, display=True):
    results = face_mesh.process(image[:, :, ::-1])
    output_image = image[:, :, ::-1].copy()
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

    if display:
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(output_image)
        plt.title("Output")
        plt.axis("off")
    else:
        return np.ascontiguousarray(output_image[:, :, ::-1], dtype=np.uint8), results


def getSize(image, face_landmarks, INDEXES):
    image_height, image_width, _ = image.shape

    INDEXES_LIST = list(itertools.chain(*INDEXES))
    landmarks = []
    for INDEX in INDEXES_LIST:

        landmarks.append(
            [
                int(face_landmarks.landmark[INDEX].x * image_width),
                int(face_landmarks.landmark[INDEX].y * image_height),
            ]
        )

    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    landmarks = np.array(landmarks)
    return width, height, landmarks


def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
    annotated_image = image.copy()
    try:
        filter_img_height, filter_img_width, _ = filter_img.shape

        _, face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)

        required_height = int(face_part_height * 3)

        resized_filter_img = cv2.resize(
            filter_img,
            (
                int(filter_img_width * (required_height / filter_img_height)),
                required_height,
            ),
        )

        filter_img_height, filter_img_width, _ = resized_filter_img.shape

        _, filter_img_mask = cv2.threshold(
            cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
            1,
            255,
            cv2.THRESH_BINARY_INV,
        )
        # filter_img_mask=image[:, :, 3]

        center = landmarks.mean(axis=0).astype("int")

        if face_part == "MOUTH":
            location = (
                int(center[0] - filter_img_width / 2.3),
                int(center[1]) - int(filter_img_height / 1.5),
            )
            # print(filter_img_height,filter_img_width)

        ROI = image[
            location[1] : location[1] + filter_img_height,
            location[0] : location[0] + filter_img_width,
        ]

        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
        resultant_image = cv2.bitwise_or(resultant_image, resized_filter_img)
        annotated_image[
            location[1] : location[1] + filter_img_height,
            location[0] : location[0] + filter_img_width,
        ] = resultant_image

    except Exception as e:
        pass
    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(annotated_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")
    else:
        return annotated_image


def apply_mustache_to_image(image_path, mustache_path):
    # Load the image and the mustache
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mustache = cv2.imread(mustache_path)

    # Detect facial landmarks
    _, face_mesh_results = detectFacialLandmarks(image, face_mesh, display=False)

    # If facial landmarks are detected
    if face_mesh_results.multi_face_landmarks:
        # For each face
        for face_num, face_landmarks in enumerate(
            face_mesh_results.multi_face_landmarks
        ):
            # Apply the mustache
            image = overlay(
                image,
                mustache,
                face_landmarks,
                "MOUTH",
                mp_face_mesh.FACEMESH_LIPS,
                display=False,
            )

    return image


if __name__ == "__main__":
    img = apply_mustache_to_image(
        "data/0.jpg", "data_augmentation/filters/mustache1.png"
    )
    cv2.imwrite("data/0_mustache.jpg", img)

    img = apply_mustache_to_image(
        "data/1.jpg", "data_augmentation/filters/mustache1.png"
    )
    cv2.imwrite("data/1_mustache.jpg", img)
    # img = apply_mustache_to_image('media/sample2.jpg', 'media/mustache1.png')
    # cv2.imwrite('media/test_2.jpg', img)
