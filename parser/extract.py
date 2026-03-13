import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v84_locked_column_classifier"

# --------------------------------------------------
# LOCKED SAMPLING COLUMN
# --------------------------------------------------

X_LEFT = 939
X_RIGHT = 951

# sampling height
BLOCK_HEIGHT = 16


# --------------------------------------------------
# ROW START COORDINATES (LOCKED)
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
# DEBUG COLOR MAP
# --------------------------------------------------

COLOR_MAP = {
    "green": (0,255,0),
    "yellow": (0,255,255),
    "orange": (0,165,255),
    "red": (0,0,255),
    None: (200,200,200)
}


# --------------------------------------------------
# COLOR CLASSIFIER (HSV)
# --------------------------------------------------

def classify_color(block):

    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:,:,0])
    s = np.mean(hsv[:,:,1])
    v = np.mean(hsv[:,:,2])

    if s < 40:
        return None

    # red
    if h < 10 or h > 170:
        return "red"

    # orange
    if 10 <= h < 22:
        return "orange"

    # yellow
    if 22 <= h < 35:
        return "yellow"

    # green
    if 35 <= h < 85:
        return "green"

    return None


# --------------------------------------------------
# MAIN EXTRACTION
# --------------------------------------------------

def extract_scores(pdf_bytes, debug=False):

    # LOCKED DPI — DO NOT CHANGE
    pages = convert_from_bytes(pdf_bytes, dpi=200)

    if len(pages) < 2:
        raise Exception("PDF missing page 2")

    page = pages[1]

    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    scores = {}

    for disease, y in ROW_START.items():

        block = img[y:y+BLOCK_HEIGHT, X_LEFT:X_RIGHT]

        if block.size == 0:
            scores[disease] = None
            continue

        result = classify_color(block)

        scores[disease] = result

        if debug:

            color = COLOR_MAP.get(result)

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

        # draw sampling column
        cv2.line(img, (X_LEFT,0), (X_LEFT,img.shape[0]), (255,0,0), 2)
        cv2.line(img, (X_RIGHT,0), (X_RIGHT,img.shape[0]), (255,0,0), 2)

        ok, png = cv2.imencode(".png", img)

        return png.tobytes()

    return {
        "engine": ENGINE_NAME,
        "scores": scores
    }
