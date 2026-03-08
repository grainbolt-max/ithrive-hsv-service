import numpy as np
import cv2

from parser.disease_list import DISEASE_LIST

# ===================================
# RISK BAR SAMPLING WINDOW
# ===================================

SCORE_SAMPLE_LEFT = 904
SCORE_SAMPLE_RIGHT = 906


# ===================================
# COLOR DETECTION
# ===================================

def detect_bar_color(region):

    if region.size == 0:
        return "grey"

    avg = region.mean(axis=(0, 1))

    r, g, b = avg

    if r > 200 and g < 120:
        return "red"

    if r > 180 and g > 120:
        return "orange"

    if g > 180:
        return "green"

    return "grey"


# ===================================
# MAIN EXTRACTION FUNCTION
# ===================================

def extract_disease_scores(page, anchors, rows):

    scores = {}

    for i, y in enumerate(rows):

        if i >= len(DISEASE_LIST):
            break

        y1 = int(y - 4)
        y2 = int(y + 4)

        region = page[y1:y2, SCORE_SAMPLE_LEFT:SCORE_SAMPLE_RIGHT]

        if region.size == 0:
            scores[DISEASE_LIST[i]] = "grey"
            continue

        color = detect_bar_color(region)

        scores[DISEASE_LIST[i]] = color

    return scores
