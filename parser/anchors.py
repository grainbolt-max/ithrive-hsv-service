import cv2
import numpy as np


def find_header_anchor(img):
    """
    Anchor 1
    Finds the first non-white row (top of report content)
    """

    h, w, _ = img.shape

    for y in range(h):
        row = img[y:y+1, :]
        if np.mean(row) < 250:
            return y

    return 0


def find_left_column_anchor(img):
    """
    Anchor 2
    Finds where disease names begin
    """

    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    column_profile = np.mean(gray, axis=0)

    for x in range(w):
        if column_profile[x] < 240:
            return x

    return 0


def find_risk_bar_anchor(img):
    """
    Anchor 3
    Finds the vertical column containing the risk bars
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 40, 40])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    column_sum = np.sum(mask, axis=0)

    x = np.argmax(column_sum)

    return int(x)


def find_table_bottom_anchor(img):
    """
    Anchor 4
    Detect bottom of disease list
    """

    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for y in range(h-1, 0, -1):

        row = gray[y:y+1, :]

        if np.mean(row) < 245:
            return y

    return h


def find_footer_anchor(img):
    """
    Anchor 5
    Detect footer separator line
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=300,
        maxLineGap=10
    )

    if lines is None:
        return None

    y_positions = [line[0][1] for line in lines]

    return max(y_positions)


def detect_all_anchors(img):
    """
    Master anchor detector
    """

    anchors = {}

    anchors["header_y"] = find_header_anchor(img)

    anchors["left_x"] = find_left_column_anchor(img)

    anchors["risk_bar_x"] = find_risk_bar_anchor(img)

    anchors["table_bottom_y"] = find_table_bottom_anchor(img)

    anchors["footer_y"] = find_footer_anchor(img)

    return anchors