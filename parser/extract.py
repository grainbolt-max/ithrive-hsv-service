import cv2
import numpy as np
from parser.disease_list import DISEASE_LIST

# Deterministic sampling stripe inside risk bars
SCORE_SAMPLE_LEFT = 790
SCORE_SAMPLE_RIGHT = 796


def detect_bar_color(bar_region):

    hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)

    # color masks
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (5, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    orange_mask = cv2.inRange(hsv, (5, 80, 80), (15, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 60, 60), (85, 255, 255))
    blue_mask = cv2.inRange(hsv, (85, 60, 60), (130, 255, 255))

    # pixel counts
    red_pixels = cv2.countNonZero(red_mask)
    orange_pixels = cv2.countNonZero(orange_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    blue_pixels = cv2.countNonZero(blue_mask)

    total_pixels = bar_region.shape[0] * bar_region.shape[1]

    if red_pixels > total_pixels * 0.01:
        return "red"

    if orange_pixels > total_pixels * 0.01:
        return "orange"

    if yellow_pixels > total_pixels * 0.01:
        return "yellow"

    if green_pixels > total_pixels * 0.01:
        return "green"

    if blue_pixels > total_pixels * 0.01:
        return "blue"

    return "grey"


def extract_disease_scores(img, anchors, rows):

    scores = {}

    # deterministic stripe instead of anchor detection
    x1 = SCORE_SAMPLE_LEFT
    x2 = SCORE_SAMPLE_RIGHT

    for i, row in enumerate(rows[:len(DISEASE_LIST)]):

        y_center = int(row)

        # sample a thin band around the row center
        y1 = max(0, y_center - 3)
        y2 = y_center + 3

        bar_region = img[y1:y2, x1:x2]

        if bar_region.size == 0:
            scores[DISEASE_LIST[i]] = "grey"
            continue

        color = detect_bar_color(bar_region)

        scores[DISEASE_LIST[i]] = color

    return scores
