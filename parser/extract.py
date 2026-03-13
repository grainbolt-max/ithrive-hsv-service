import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v72_blue_intensity_classifier"

# Exact bar column location at dpi=200
X_LEFT = 937

# Vertical scan range for disease bars
MIN_Y = 880
MAX_Y = 2050

DISEASES = [
"large_artery_stiffness",
"peripheral_vessel",
"blood_pressure_uncontrolled",
"small_medium_artery_stiffness",
"atherosclerosis",
"ldl_cholesterol",
"lv_hypertrophy",
"diabetes",
"metabolic_syndrome",
"insulin_resistance",
"beta_cell_function_decreased",
"blood_glucose_uncontrolled",
"tissue_inflammatory_process",
"hypothyroidism",
"hyperthyroidism",
"hepatic_fibrosis",
"chronic_hepatitis",
"prostate_cancer",
"respiratory_disorders",
"kidney_function_disorders",
"digestive_disorders",
"major_depression",
"adhd_children_learning",
"cerebral_dopamine_decreased",
"cerebral_serotonin_decreased"
]

# ------------------------------------------------
# ROW DETECTION
# ------------------------------------------------
def detect_rows(img):

    DETECT_X = X_LEFT - 10

    column = img[MIN_Y:MAX_Y, DETECT_X:DETECT_X + 40]

    gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)

    rows = []
    inside = False
    start = 0

    for y in range(gray.shape[0]):

        intensity = np.mean(gray[y,:])

        if intensity < 225 and not inside:
            start = y
            inside = True

        if intensity > 240 and inside:

            end = y

            if 10 < (end-start) < 40:
                rows.append((start + MIN_Y, end + MIN_Y))

            inside = False

    filtered = []

    for r in rows:

        if not filtered or r[0] - filtered[-1][0] > 18:
            filtered.append(r)

    return filtered


# ------------------------------------------------
# SAMPLE BAR COLOR
# ------------------------------------------------
def sample_bar_color(img, y1, y2):

    mid = int((y1 + y2) / 2)

    bar_zone = img[mid-3:mid+3, X_LEFT-80:X_LEFT+50]

    hsv = cv2.cvtColor(bar_zone, cv2.COLOR_BGR2HSV)

    for x in range(hsv.shape[1]):

        if hsv[0,x,1] > 40:

            h = int(hsv[0,x,0])
            s = int(hsv[0,x,1])
            v = int(hsv[0,x,2])

            return np.array([h,s,v])

    return np.array([0,0,255])


# ------------------------------------------------
# CLASSIFY BAR COLOR
# ------------------------------------------------
def classify_bar(sample):

    h, s, v = sample

    # ignore background / grey
    if s < 30:
        return "grey"

    # darker blue = higher risk
    if v < 110:
        return "red"

    if 110 <= v < 170:
        return "orange"

    if v >= 170:
        return "yellow"

    return "grey"


# ------------------------------------------------
# DEBUG OVERLAY
# ------------------------------------------------
def draw_debug(img, rows, scores):

    debug = img.copy()

    colors = {
        "grey": (128,128,128),
        "yellow": (0,255,255),
        "orange": (0,165,255),
        "red": (0,0,255)
    }

    for i,(y1,y2) in enumerate(rows):

        if i >= len(DISEASES):
            break

        risk = scores.get(DISEASES[i],"grey")

        cv2.rectangle(
            debug,
            (X_LEFT,y1),
            (X_LEFT+18,y2),
            colors[risk],
            4
        )

    return debug


# ------------------------------------------------
# MAIN PARSER
# ------------------------------------------------
def parse_report(pdf_bytes, debug=False):

    images = convert_from_bytes(pdf_bytes, dpi=200)

    img = np.array(images[1])

    rows = detect_rows(img)

    samples = []

    for y1,y2 in rows:
        samples.append(sample_bar_color(img,y1,y2))

    scores = {}

    for i,(y1,y2) in enumerate(rows):

        if i >= len(DISEASES):
            break

        scores[DISEASES[i]] = classify_bar(samples[i])

    if debug:

        overlay = draw_debug(img, rows, scores)

        _,png = cv2.imencode(".png",overlay)

        return png.tobytes()

    return {
        "engine": ENGINE_NAME,
        "scores": scores
    }


def extract_scores(pdf_bytes):

    return parse_report(pdf_bytes)
