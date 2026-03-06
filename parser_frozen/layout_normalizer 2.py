import cv2

TARGET_WIDTH = 1654


def normalize_dpi(img):

    h, w = img.shape[:2]

    scale = TARGET_WIDTH / w

    if abs(scale - 1.0) < 0.05:
        return img, 1.0

    resized = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR
    )

    return resized, scale