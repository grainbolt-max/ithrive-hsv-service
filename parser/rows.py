import cv2
import numpy as np

def detect_row_lines(img, top_y, bottom_y):
    region = img[top_y:bottom_y, :]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    horizontal_strength = np.sum(edges, axis=1)
    candidates = np.where(horizontal_strength > 200)[0]

    rows = []
    last = -50

    for y in candidates:
        if y - last > 18:
            rows.append(int(y + top_y))
            last = y

    return rows
