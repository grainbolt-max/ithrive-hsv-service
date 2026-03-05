import cv2
import numpy as np

def detect_table_divider(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,50,150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=400,
        maxLineGap=10
    )

    candidates = []

    if lines is not None:

        for line in lines:

            x1,y1,x2,y2 = line[0]

            if abs(x1-x2) < 5:

                length = abs(y2-y1)

                if length > 600:

                    candidates.append(x1)

    if len(candidates)==0:
        return None

    return int(np.median(candidates))


def find_first_nonwhite_row(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray.shape

    for y in range(h):

        if np.mean(gray[y]) < 250:

            return y

    return 0


def detect_left_column(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,50,150)

    vertical_strength = np.sum(edges,axis=0)

    candidates = np.where(vertical_strength>200)[0]

    if len(candidates)==0:

        return 150

    return int(candidates[0])


def detect_footer(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray.shape

    for y in range(h-1,0,-1):

        if np.mean(gray[y]) < 250:

            return y

    return None


def detect_all_anchors(img):

    anchors = {}

    header_y = find_first_nonwhite_row(img)

    left_x = detect_left_column(img)

    footer_y = detect_footer(img)

    anchors["header_y"] = header_y
    anchors["left_x"] = left_x
    anchors["footer_y"] = footer_y

    divider_x = detect_table_divider(img)

    if divider_x is not None:

        anchors["left_x"] = divider_x - 190
        anchors["risk_bar_x"] = divider_x + 20

    else:

        anchors["risk_bar_x"] = anchors["left_x"] + 200

    anchors["table_bottom_y"] = footer_y - 1 if footer_y is not None else img.shape[0]-1

    return anchors
