import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v75_locked_column_classifier"

# --------------------------------------------
# LOCKED COLOR SAMPLING COLUMN
# --------------------------------------------

X_LEFT = 941
X_RIGHT = 951


# --------------------------------------------
# DISEASE ROW COORDINATES
# --------------------------------------------

ROW_MAP = {

    # PANEL 1
    "large_artery_stiffness": (908, 933),
    "peripheral_vessel": (958, 983),
    "blood_pressure_uncontrolled": (1008, 1033),
    "small_medium_artery_stiffness": (1058, 1083),
    "atherosclerosis": (1108, 1133),
    "ldl_cholesterol": (1158, 1183),
    "lv_hypertrophy": (1208, 1233),
    "metabolic_syndrome": (1283, 1308),
    "insulin_resistance": (1333, 1358),
    "beta_cell_function_decreased": (1383, 1408),
    "blood_glucose_uncontrolled": (1433, 1458),
    "tissue_inflammatory_process": (1483, 1508),

    # PANEL 2
    "hypothyroidism": (1490, 1520),
    "hyperthyroidism": (1540, 1565),
    "hepatic_fibrosis": (1590, 1620),
    "chronic_hepatitis": (1630, 1655),
    "prostate_cancer": (1675, 1700),
    "respiratory_disorders": (1725, 1750),
    "kidney_function_disorders": (1775, 1800),
    "digestive_disorders": (1825, 1850),
    "major_depression": (1920, 1945),
    "adhd_children_learning": (1960, 1985),
    "cerebral_dopamine_decreased": (2015, 2040),
    "cerebral_serotonin_decreased": (2050, 2075),
}


# --------------------------------------------
# COLOR CLASSIFICATION
# --------------------------------------------

def classify_color(region):

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    avg = hsv.mean(axis=(0,1))
    h, s, v = avg

    # red
    if (h < 10 or h > 170) and s > 120 and v > 120:
        return "red"

    # orange
    if 10 < h < 25 and s > 120:
        return "orange"

    # yellow
    if 25 < h < 40 and s > 120:
        return "yellow"

    return None


# --------------------------------------------
# MAIN PARSER
# --------------------------------------------

def extract_scores(pdf_bytes, debug=False):

    images = convert_from_bytes(pdf_bytes)

    page = np.array(images[1])
    page = cv2.cvtColor(page, cv2.COLOR_RGB2BGR)

    results = {}

    debug_img = page.copy()

    for disease, (y1, y2) in ROW_MAP.items():

        region = page[y1:y2, X_LEFT:X_RIGHT]

        color = classify_color(region)

        results[disease] = color

        if debug:

            cv2.rectangle(
                debug_img,
                (X_LEFT, y1),
                (X_RIGHT, y2),
                (0,255,0),
                2
            )

    if debug:

        cv2.line(
            debug_img,
            (X_LEFT, 0),
            (X_LEFT, debug_img.shape[0]),
            (255,0,0),
            2
        )

        _, buffer = cv2.imencode(".png", debug_img)
        return buffer.tobytes()

    return {
        "engine": ENGINE_NAME,
        "scores": results
    }
