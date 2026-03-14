import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v73_blue_intensity_classifier_fixed"

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
# DETECT ROW POSITIONS
# ------------------------------------------------
def detect_rows(img):

    column = img[MIN_Y:MAX_Y, X_LEFT:X_LEFT+15]

    hsv = cv2.cvtColor(column, cv2.COLOR_BGR2HSV)

    rows = []

    inside = False
    start = 0

    for y in range(hsv.shape[0]):

        s = np.mean(hsv[y,:,1])

        if s > 40 and not inside:
            start = y
            inside = True

        if s < 20 and inside:

            end = y

            if 10 < (end-start) < 40:
                rows.append((start+MIN_Y,end+MIN_Y))

            inside = False

    # remove duplicates
    filtered = []

    for r in rows:

        if not filtered:
            filtered.append(r)
            continue

        if r[0] - filtered[-1][0] > 18:
            filtered.append(r)

    return filtered


# ------------------------------------------------
# SAMPLE BAR COLOR
# ------------------------------------------------
def sample_bar_color(img, y1, y2):

    mid = int((y1+y2)/2)

    sample = img[mid-2:mid+2, X_LEFT:X_LEFT+15]

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:,:,0])
    s = np.mean(hsv[:,:,1])
    v = np.mean(hsv[:,:,2])

    return np.array([h,s,v])


# ------------------------------------------------
# CLASSIFY BAR COLOR
# ------------------------------------------------
def classify_bar(sample):

    h, s, v = sample

    # ignore background
    if s < 20:
        return None

    # dark blue = highest risk
    if v < 120:
        return "red"

    # medium blue
    if v < 175:
        return "orange"

    # cyan/light blue baseline
    return "yellow"


# ------------------------------------------------
# DEBUG DRAW
# ------------------------------------------------
def draw_debug(img, rows, scores):

    debug = img.copy()

    colors = {
        "yellow":(0,255,255),
        "orange":(0,165,255),
        "red":(0,0,255)
    }

    for i,(y1,y2) in enumerate(rows):

        if i >= len(DISEASES):
            break

        risk = scores.get(DISEASES[i])

        if risk is None:
            continue

        cv2.rectangle(
            debug,
            (X_LEFT,y1),
            (X_LEFT+15,y2),
            colors[risk],
            3
        )

    return debug


# ------------------------------------------------
# MAIN PARSER
# ------------------------------------------------
def parse_report(pdf_bytes, debug=False):

    images = convert_from_bytes(pdf_bytes, dpi=200)

    img = np.array(images[1])

    rows = detect_rows(img)

    scores = {}

    for i,(y1,y2) in enumerate(rows):

        if i >= len(DISEASES):
            break

        sample = sample_bar_color(img,y1,y2)

        scores[DISEASES[i]] = classify_bar(sample)

    if debug:

        overlay = draw_debug(img,rows,scores)

        _,png = cv2.imencode(".png",overlay)

        return png.tobytes()

    return {
        "engine":ENGINE_NAME,
        "scores":scores
    }


def extract_scores(pdf_bytes):

    return parse_report(pdf_bytes)
