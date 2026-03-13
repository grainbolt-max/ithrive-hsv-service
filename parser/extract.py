import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v76_locked_column_classifier"

# Locked sampling column
X_LEFT = 939
X_RIGHT = 951

# --------------------------------------------------
# DISEASE ROW COORDINATES (shifted to match panel)
# --------------------------------------------------

ROW_MAP = {

    # PANEL 1
    "large_artery_stiffness": (910, 935),
    "peripheral_vessel": (960, 985),
    "blood_pressure_uncontrolled": (1010, 1035),
    "small_medium_artery_stiffness": (1060, 1085),
    "atherosclerosis": (1110, 1135),
    "ldl_cholesterol": (1160, 1185),
    "lv_hypertrophy": (1210, 1235),
    "metabolic_syndrome": (1285, 1310),
    "insulin_resistance": (1335, 1360),
    "beta_cell_function_decreased": (1385, 1410),
    "blood_glucose_uncontrolled": (1435, 1460),
    "tissue_inflammatory_process": (1485, 1510),

    # PANEL 2
    "hypothyroidism": (1825, 1855),
    "hyperthyroidism": (1875, 1900),
    "hepatic_fibrosis": (1925, 1955),
    "chronic_hepatitis": (1965, 1990),
    "prostate_cancer": (2010, 2035),
    "respiratory_disorders": (2060, 2085),
    "kidney_function_disorders": (2110, 2135),
    "digestive_disorders": (2160, 2185),
    "major_depression": (2255, 2280),
    "adhd_children_learning": (2295, 2320),
    "cerebral_dopamine_decreased": (2350, 2375),
    "cerebral_serotonin_decreased": (2385, 2410),
}


# --------------------------------------------------
# COLOR CLASSIFIER
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
    img = np.array(pages[1])

    scores = {}

    debug_img = img.copy()

    height = img.shape[0]

    # Draw sampling column
    if debug:
        cv2.rectangle(
            debug_img,
            (X_LEFT, 0),
            (X_RIGHT, height),
            (255, 0, 0),
            2
        )

    for disease, (y1, y2) in ROW_MAP.items():

        region = img[y1:y2, X_LEFT:X_RIGHT]

        mean_color = np.mean(region.reshape(-1, 3), axis=0)

        classification = classify_color(mean_color)

        scores[disease] = classification

        if debug:

            # green rectangle for sampled area
            cv2.rectangle(
                debug_img,
                (X_LEFT, y1),
                (X_RIGHT, y2),
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
