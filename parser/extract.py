import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v72_offset_row_classifier"

# ------------------------------------------------
# BAR SAMPLING COLUMN
# ------------------------------------------------

X_LEFT = 939
X_RIGHT = 954

# ------------------------------------------------
# ROW STRUCTURE
# ------------------------------------------------

FIRST_ROW = 882

ROW_STEPS = [
0, 44, 87, 130, 172,
214, 255, 297, 338, 380,
421, 462, 532, 573, 615,
656, 698, 739, 781, 822,
864, 905, 947, 988
]

ROWS = [FIRST_ROW + step for step in ROW_STEPS]

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
# SAMPLE COLOR
# ------------------------------------------------

def sample_bar_color(img, y):

    crop = img[y-6:y+6, X_LEFT:X_RIGHT]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:,:,0])
    s = np.mean(hsv[:,:,1])
    v = np.mean(hsv[:,:,2])

    return h,s,v


# ------------------------------------------------
# CLASSIFY COLOR
# ------------------------------------------------

def classify_bar(h,s,v):

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

    for y in ROWS:

        cv2.rectangle(
            overlay,
            (X_LEFT-10, y-8),
            (X_RIGHT+10, y+8),
            (0,255,0),
            2
        )

    return overlay


# ------------------------------------------------
# MAIN PARSER
# ------------------------------------------------

def parse_report(pdf_bytes, debug=False):

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

        h,s,v = color

        scores[disease] = classify_bar(h,s,v)

    if debug:

        overlay = draw_debug(img)

        ok,png = cv2.imencode(".png",overlay)

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
