import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v87_locked_column_bar_classifier"

# --------------------------------------------------
# LOCKED SAMPLING COLUMN
# --------------------------------------------------

X_LEFT = 939
X_RIGHT = 951
BLOCK_HEIGHT = 16

# --------------------------------------------------
# LOCKED ROW COORDINATES
# --------------------------------------------------

ROW_START = {

    # PANEL 1
    "large_artery_stiffness": 920,
    "peripheral_vessel": 950,
    "blood_pressure_uncontrolled": 985,
    "small_medium_artery_stiffness": 1015,
    "atherosclerosis": 1050,
    "ldl_cholesterol": 1075,
    "lv_hypertrophy": 1110,
    "metabolic_syndrome": 1170,
    "insulin_resistance": 1200,
    "beta_cell_function_decreased": 1235,
    "blood_glucose_uncontrolled": 1265,
    "tissue_inflammatory_process": 1295,

    # PANEL 2
    "hypothyroidism": 1530,
    "hyperthyroidism": 1545,
    "hepatic_fibrosis": 1590,
    "chronic_hepatitis": 1620,
    "prostate_cancer": 1655,
    "respiratory_disorders": 1685,
    "kidney_function_disorders": 1715,
    "digestive_disorders": 1745,
    "major_depression": 1810,
    "adhd_children_learning": 1840,
    "cerebral_dopamine_decreased": 1870,
    "cerebral_serotonin_decreased": 1910,
}

# --------------------------------------------------
# DEBUG COLORS
# --------------------------------------------------

COLOR_MAP = {
    "yellow": (0,255,255),
    "orange": (0,165,255),
    "red": (0,0,255),
    None: (160,160,160)
}

# --------------------------------------------------
# SAMPLE BAR COLOR
# --------------------------------------------------

def sample_bar_color(img, y):

    block = img[y:y+BLOCK_HEIGHT, X_LEFT:X_RIGHT]

    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)

    h = hsv[:,:,0]
    s = hsv[:,:,1]

    mean_h = np.mean(h)
    mean_s = np.mean(s)

    # --------------------------------------------------
    # FILTER TEXT / GRAY ROWS
    # --------------------------------------------------

    if mean_s < 60:
        return None

    # remove rows with mixed pixels (text edges)
    if np.std(h) > 8:
        return None

    return mean_h

# --------------------------------------------------
# CLASSIFY COLOR
# --------------------------------------------------

def classify_color(h):

    if h is None:
        return None

    # these ranges match the bar colors in the report
    if h < 25:
        return "yellow"

    if 25 <= h < 45:
        return "orange"

    if h >= 45:
        return "red"

    return None

# --------------------------------------------------
# MAIN EXTRACTION
# --------------------------------------------------

def extract_scores(pdf_bytes, debug=False):

    # DPI LOCKED — DO NOT CHANGE
    pages = convert_from_bytes(pdf_bytes, dpi=200)

    if len(pages) < 2:
        raise Exception("PDF missing page 2")

    page = pages[1]

    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    scores = {}

    for disease, y in ROW_START.items():

        hue = sample_bar_color(img, y)

        risk = classify_color(hue)

        scores[disease] = risk

        if debug:

            color = COLOR_MAP.get(risk)

            cv2.rectangle(
                img,
                (X_LEFT, y),
                (X_RIGHT, y + BLOCK_HEIGHT),
                color,
                2
            )

            cv2.putText(
                img,
                disease,
                (X_RIGHT + 8, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255,255,255),
                1
            )

    if debug:

        cv2.line(img, (X_LEFT,0), (X_LEFT,img.shape[0]), (255,0,0), 2)
        cv2.line(img, (X_RIGHT,0), (X_RIGHT,img.shape[0]), (255,0,0), 2)

        ok, png = cv2.imencode(".png", img)

        return png.tobytes()

    return {
        "engine": ENGINE_NAME,
        "scores": scores
    }
