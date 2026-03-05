import cv2
import numpy as np
import pytesseract
from parser.disease_list import DISEASES


def read_disease_name(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7"
    )

    text = text.strip()

    return text


def measure_risk_bar(bar_img):

    hsv = cv2.cvtColor(bar_img, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(hsv,(15,80,80),(40,255,255))
    mask_orange = cv2.inRange(hsv,(5,80,80),(15,255,255))
    mask_red = cv2.inRange(hsv,(0,80,80),(5,255,255))

    mask = mask_yellow | mask_orange | mask_red

    column_activity = np.sum(mask,axis=0)

    active_columns = np.where(column_activity>0)[0]

    if len(active_columns)==0:
        return 0.0

    width = active_columns[-1] - active_columns[0]

    return float(width) / float(bar_img.shape[1])


def extract_disease_scores(img, rows, anchors):

    diseases = {}

    text_left = anchors["left_x"] + 20
    text_right = anchors["risk_bar_x"] - 20

    bar_left = anchors["risk_bar_x"] + 10
    bar_right = bar_left + 300

    for i in range(len(rows)-1):

        y1 = rows[i]
        y2 = rows[i+1]

        row = img[y1:y2,:]

        text_region = row[:,text_left:text_right]

        bar_region = row[:,bar_left:bar_right]

        score = measure_risk_bar(bar_region)

        if score < 0.05:
            continue

        name = DISEASES[i] if i < len(DISEASES) else f"row_{i}"

        diseases[name] = score

    return diseases