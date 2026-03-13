import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v69_scaled_coordinate_classifier"

# ---------------------------------------------------
# DPI CONFIG
# ---------------------------------------------------

BASE_DPI = 300
CURRENT_DPI = 200

SCALE = CURRENT_DPI / BASE_DPI

# ---------------------------------------------------
# LOCKED SAMPLING COLUMN (scaled from 300dpi)
# ---------------------------------------------------

X_LEFT = int(939 * SCALE)
X_RIGHT = int(954 * SCALE)

# ---------------------------------------------------
# DETECTION WINDOW
# ---------------------------------------------------

MIN_Y = int(880 * SCALE)
MAX_Y = int(2050 * SCALE)

ROW_HEIGHT = int(46 * SCALE)

# ---------------------------------------------------
# DISEASE ORDER
# ---------------------------------------------------

DISEASES = [
    "large_artery_stiffness",
    "peripheral_vessel",
    "blood_pressure_uncontrolled",
    "small_medium_artery_stiffness",
    "atherosclerosis",
    "ldl_cholesterol",
    "lv_hypertrophy",
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
    "cerebral_serotonin_decreased",
]

# ---------------------------------------------------
# ROW DETECTION
# ---------------------------------------------------

def detect_rows():

    rows = []
    y = MIN_Y

    for _ in range(len(DISEASES)):
        rows.append(int(y))
        y += ROW_HEIGHT

    return rows


# ---------------------------------------------------
# COLOR SAMPLING
# ---------------------------------------------------

def sample_bar_color(img, y):

    crop = img[y - 6:y + 6, X_LEFT:X_RIGHT]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])

    return h, s, v


# ---------------------------------------------------
# COLOR CLASSIFIER
# ---------------------------------------------------

def classify_bar(h, s, v):

    if s < 40:
        return None

    if h < 10:
        return "red"

    if h < 25:
        return "orange"

    if h < 45:
        return "yellow"

    return None


# ---------------------------------------------------
# DEBUG OVERLAY
# ---------------------------------------------------

def draw_debug(img, rows, scores):

    overlay = img.copy()

    # sampling column
    cv2.rectangle(
        overlay,
        (X_LEFT, MIN_Y),
        (X_RIGHT, MAX_Y),
        (255, 0, 0),
        2
    )

    # row markers
    for y in rows:

        cv2.rectangle(
            overlay,
            (X_LEFT - 10, y - 8),
            (X_RIGHT + 10, y + 8),
            (0, 255, 0),
            1
        )

    return overlay


# ---------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------

def parse_report(pdf_bytes, debug=False):

    pages = convert_from_bytes(pdf_bytes, dpi=CURRENT_DPI)

    if len(pages) < 2:
        raise Exception("PDF missing disease screening page")

    page = pages[1]

    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    rows = detect_rows()

    scores = {}

    for disease, y in zip(DISEASES, rows):

        color = sample_bar_color(img, y)

        if color is None:
            scores[disease] = None
            continue

        h, s, v = color

        scores[disease] = classify_bar(h, s, v)

    if debug:

        overlay = draw_debug(img, rows, scores)

        ok, png = cv2.imencode(".png", overlay)

        return png.tobytes()

    ordered_scores = {d: scores.get(d) for d in DISEASES}

    return {
        "engine": ENGINE_NAME,
        "scores": ordered_scores
    }


# ---------------------------------------------------
# EXTERNAL ENTRYPOINT
# ---------------------------------------------------

def extract_scores(pdf_bytes, debug=False):
    return parse_report(pdf_bytes, debug=debug)
