import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v74_locked_column_classifier"

# ------------------------------------------------
# LOCKED COLOR SAMPLING COLUMN
# ------------------------------------------------

X_LEFT = 942
X_RIGHT = 951

# ------------------------------------------------
# ROW CENTERS (derived from your measured ranges)
# ------------------------------------------------

ROWS = [
1387, 1437, 1487, 1537, 1587, 1637,
1687, 1762, 1812, 1862, 1912, 1962,
2305, 2352, 2405, 2442, 2487, 2537,
2587, 2637, 2732, 2772, 2827, 2862
]

# ------------------------------------------------
# DISEASE ORDER
# ------------------------------------------------

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
"cerebral_serotonin_decreased"
]

# ------------------------------------------------
# SAMPLE COLOR FROM BAR
# ------------------------------------------------

def sample_bar_color(img, y):

    crop = img[y-6:y+6, X_LEFT:X_RIGHT]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:,:,0])
    s = np.mean(hsv[:,:,1])
    v = np.mean(hsv[:,:,2])

    return h, s, v


# ------------------------------------------------
# CLASSIFY COLOR
# ------------------------------------------------

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


# ------------------------------------------------
# DEBUG OVERLAY
# ------------------------------------------------

def draw_debug(img):

    overlay = img.copy()

    # draw sampling column
    cv2.rectangle(
        overlay,
        (X_LEFT, 0),
        (X_RIGHT, img.shape[0]),
        (255,0,0),
        2
    )

    # draw row markers
    for y in ROWS:

        cv2.rectangle(
            overlay,
            (X_LEFT-8, y-8),
            (X_RIGHT+8, y+8),
            (0,255,0),
            2
        )

    return overlay


# ------------------------------------------------
# MAIN PARSER
# ------------------------------------------------

def parse_report(pdf_bytes, debug=False):

    # IMPORTANT: 300 DPI so coordinates match measured values
    pages = convert_from_bytes(pdf_bytes, dpi=200)

    if len(pages) < 2:
        raise Exception("PDF missing disease screening page")

    page = pages[1]

    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    scores = {}

    for disease, y in zip(DISEASES, ROWS):

        color = sample_bar_color(img, y)

        if color is None:
            scores[disease] = None
            continue

        h, s, v = color

        scores[disease] = classify_bar(h, s, v)

    if debug:

        overlay = draw_debug(img)

        ok, png = cv2.imencode(".png", overlay)

        return png.tobytes()

    ordered_scores = {d: scores.get(d) for d in DISEASES}

    return {
        "engine": ENGINE_NAME,
        "scores": ordered_scores
    }


# ------------------------------------------------
# EXTERNAL ENTRYPOINT
# ------------------------------------------------

def extract_scores(pdf_bytes, debug=False):
    return parse_report(pdf_bytes, debug=debug)
