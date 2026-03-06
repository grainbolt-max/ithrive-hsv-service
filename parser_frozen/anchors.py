import cv2
import numpy as np

def detect_all_anchors(img):

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    scan_start = int(w * 0.45)
    scan_end = int(w * 0.75)

    roi = edges[:, scan_start:scan_end]

    column_strength = np.sum(roi, axis=0)

    peak_column = int(np.argmax(column_strength))

    risk_bar_x = scan_start + peak_column

    risk_bar_width = int(w * 0.14)

    anchors = {}

    anchors["risk_bar_x"] = int(risk_bar_x)
    anchors["risk_bar_width"] = int(risk_bar_width)

    return anchors
