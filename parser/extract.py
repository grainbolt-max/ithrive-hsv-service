import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v77_locked_column_classifier"

# LOCKED SAMPLING COLUMN
X_LEFT = 939
X_RIGHT = 951

# --------------------------------------------------
# DISEASE ROW COORDINATES
# (reduced sampling height for accurate color read)
# --------------------------------------------------

ROW_MAP = {

    # PANEL 1
    "large_artery_stiffness": (911, 927),
    "peripheral_vessel": (961, 977),
    "blood_pressure_uncontrolled": (1011, 1027),
    "small_medium_artery_stiffness": (1061, 1077),
    "atherosclerosis": (1111, 1127),
    "ldl_cholesterol": (1161, 1177),
    "lv_hypertrophy": (1211, 1227),
    "metabolic_syndrome": (1286, 1302),
    "insulin_resistance": (1336, 1352),
    "beta_cell_function_decreased": (1386, 1402),
    "blood_glucose_uncontrolled": (1436, 1452),
    "tissue_inflammatory_process": (1486, 1502),

    # PANEL 2
    "hypothyroidism": (1561, 1577),
    "hyperthyroidism": (1611, 1627),
    "hepatic_fibrosis": (1661, 1677),
    "chronic_hepatitis": (1701, 1717),
    "prostate_cancer": (1746, 1762),
    "respiratory_disorders": (1796, 1812),
    "kidney_function_disorders": (1846, 1862),
    "digestive_disorders": (1896, 1912),
    "major_depression": (1991, 2007),
    "adhd_children_learning": (2031, 2047),
    "cerebral_dopamine_decreased": (2086, 2102),
    "cerebral_serotonin_decreased": (2121, 2137),
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

    # PAGE 2 contains disease bars
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

    for disease, (y1, y2) in ROW_MAP.items():

        region = img[y1:y2, X_LEFT:X_RIGHT]

        mean_color = np.mean(region.reshape(-1, 3), axis=0)

        classification = classify_color(mean_color)

        scores[disease] = classification

        if debug:
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
