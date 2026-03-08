import cv2
import numpy as np
from parser.disease_list import DISEASE_LIST

def detect_bar_color(bar_region):

    hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, (15,80,80), (40,255,255))
    orange_mask = cv2.inRange(hsv, (5,80,80), (15,255,255))
    red_mask1 = cv2.inRange(hsv, (0,80,80), (5,255,255))
    red_mask2 = cv2.inRange(hsv, (170,80,80), (180,255,255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_pixels = cv2.countNonZero(yellow_mask)
    orange_pixels = cv2.countNonZero(orange_mask)
    red_pixels = cv2.countNonZero(red_mask)

    total_pixels = bar_region.shape[0] * bar_region.shape[1]

    if red_pixels > total_pixels * 0.01:
        return "red"

    if orange_pixels > total_pixels * 0.01:
        return "orange"

    if yellow_pixels > total_pixels * 0.01:
        return "yellow"

    return "grey"


def extract_disease_scores(img, anchors, rows):

    scores = {}

    x1 = anchors["risk_bar_x"]
    x2 = anchors["risk_bar_x"] + anchors["risk_bar_width"]

    for i, row in enumerate(rows[:len(DISEASE_LIST)]):

        y1 = int(row)
        y2 = int(row + 20)

        bar_region = img[y1:y2, x1:x2]

        color = detect_bar_color(bar_region)

        scores[DISEASE_LIST[i]] = color

    return scores