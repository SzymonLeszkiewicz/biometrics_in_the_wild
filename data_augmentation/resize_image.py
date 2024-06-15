import cv2


def resize_image(image_path, scaler=6):
    image = cv2.imread(image_path)
    image = cv2.resize(
        image,
        (image.shape[1] // scaler, image.shape[0] // scaler),
        interpolation=cv2.INTER_CUBIC,
    )
    return image
