import cv2
import numpy as np
from parser.disease_list import DISEASE_LIST

# fixed sampling window (very narrow stripe)
SCORE_SAMPLE_LEFT = 904
SCORE_SAMPLE_RIGHT = 910


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

    # use deterministic sampling stripe instead of anchor detection
    x1 = SCORE_SAMPLE_LEFT
    x2 = SCORE_SAMPLE_RIGHT

    for i, row in enumerate(rows[:len(DISEASE_LIST)]):

        # sample a slightly thicker vertical region around row center
        y_center = int(row)

        y1 = max(0, y_center - 3)
        y2 = y_center + 3

        bar_region = img[y1:y2, x1:x2]

        color = detect_bar_color(bar_region)

        scores[DISEASE_LIST[i]] = color

    return scores
