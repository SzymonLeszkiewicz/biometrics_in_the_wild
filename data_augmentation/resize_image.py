import cv2


def resize_image(image_path, scaler=6):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        raise ValueError("Image path must be a string")
    image = cv2.resize(
        image,
        (image.shape[1] // scaler, image.shape[0] // scaler),
        interpolation=cv2.INTER_CUBIC,
    )
    return image
