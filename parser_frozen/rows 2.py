import cv2
import numpy as np


def detect_rows(img, anchors):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=500,
        maxLineGap=15
    )

    rows = []

    if lines is None:
        return rows

    for line in lines:

        x1, y1, x2, y2 = line[0]

        if abs(y1 - y2) < 5:
            rows.append(int((y1 + y2) / 2))

    rows = sorted(rows)

    # remove duplicates / near rows
    filtered = []
    min_spacing = 25

    for y in rows:
        if not filtered:
            filtered.append(y)
        elif abs(y - filtered[-1]) > min_spacing:
            filtered.append(y)

    return filtered