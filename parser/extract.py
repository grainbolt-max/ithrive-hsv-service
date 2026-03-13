import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v80_locked_column_classifier"

# --------------------------------------------------
# LOCKED SAMPLING COLUMN
# --------------------------------------------------

X_LEFT = 939
X_RIGHT = 951

# height of green sampling block
BLOCK_HEIGHT = 16


# --------------------------------------------------
# ROW START COORDINATES (UPDATED)
# --------------------------------------------------

ROW_START = {

    # PANEL 1
    "large_artery_stiffness": 920,
    "peripheral_vessel": 950,
    "blood_pressure_uncontrolled": 985,
    "small_medium_artery_stiffness": 1020,
    "atherosclerosis": 1055,
    "ldl_cholesterol": 1075,
    "lv_hypertrophy": 1110,
    "metabolic_syndrome": 1170,
    "insulin_resistance": 1200,
    "beta_cell_function_decreased": 1235,
    "blood_glucose_uncontrolled": 1265,
    "tissue_inflammatory_process": 1305,

    # PANEL 2
    "hypothyroidism": 1530,
    "hyperthyroidism": 1560,
    "hepatic_fibrosis": 1590,
    "chronic_hepatitis": 1625,
    "prostate_cancer": 1655,
    "respiratory_disorders": 1685,
    "kidney_function_disorders": 1715,
    "digestive_disorders": 1745,
    "major_depression": 1805,
    "adhd_children_learning": 1840,
    "cerebral_dopamine_decreased": 1870,
    "cerebral_serotonin_decreased": 1910,
}


# --------------------------------------------------
# COLOR CLASSIFICATION
# --------------------------------------------------

def classify_color(mean_bgr):

    b, g, r = mean_bgr

    if r > 180 and g < 120:
        return "red"

    if r > 180 and g > 140:
        return "orange"

    if g > 160 and r < 150:
        return "green"

    if r > 200 and g > 200:
        return "yellow"

    return None


# --------------------------------------------------
# CORE EXTRACTION
# --------------------------------------------------

def extract_scores(pdf_bytes, debug=False):

    pages = convert_from_bytes(pdf_bytes, dpi=200)

    # disease panels live on page 2
    img = np.array(pages[1])

    scores = {}

    debug_img = img.copy()

    height = img.shape[0]

    # draw locked sampling column
    if debug:
        cv2.rectangle(
            debug_img,
            (X_LEFT, 0),
            (X_RIGHT, height),
            (255, 0, 0),
            2
        )

    for disease, start_y in ROW_START.items():

        end_y = start_y + BLOCK_HEIGHT

        region = img[start_y:end_y, X_LEFT:X_RIGHT]

        mean_color = np.mean(region.reshape(-1, 3), axis=0)

        classification = classify_color(mean_color)

        scores[disease] = classification

        if debug:
            cv2.rectangle(
                debug_img,
                (X_LEFT, start_y),
                (X_RIGHT, end_y),
                (0, 255, 0),
                2
            )

    if debug:
        _, buffer = cv2.imencode(".png", debug_img)
        return buffer.tobytes()

    return {
        "engine": ENGINE_NAME,
        "scores": scores
    }
