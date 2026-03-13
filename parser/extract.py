import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v86_locked_column_kmeans_classifier"

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

    h = np.mean(hsv[:,:,0])
    s = np.mean(hsv[:,:,1])
    v = np.mean(hsv[:,:,2])

    # ignore text rows / gray bars
    if s < 50:
        return None

    return np.array([h,s,v])


# --------------------------------------------------
# CALIBRATE COLORS USING KMEANS
# --------------------------------------------------

def calibrate_colors(samples):

    colored = np.array([s for s in samples if s is not None])

    if len(colored) < 3:
        return None, None

    data = colored[:,0].reshape(-1,1).astype(np.float32)

    K = 3

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        data,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = centers.flatten()

    order = np.argsort(centers)

    cluster_map = {
        order[0]: "yellow",
        order[1]: "orange",
        order[2]: "red"
    }

    return centers, cluster_map


# --------------------------------------------------
# CLASSIFY BAR
# --------------------------------------------------

def classify_bar(sample, centers, cluster_map):

    if sample is None:
        return None

    if centers is None:
        return None

    h = sample[0]

    dists = [abs(h - c) for c in centers]

    cluster = np.argmin(dists)

    return cluster_map[cluster]


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

    disease_list = list(ROW_START.keys())

    samples = []

    # --------------------------------------------------
    # SAMPLE ALL BARS
    # --------------------------------------------------

    for disease in disease_list:

        y = ROW_START[disease]

        hsv = sample_bar_color(img, y)

        samples.append(hsv)

    # --------------------------------------------------
    # CALIBRATE COLORS
    # --------------------------------------------------

    centers, cluster_map = calibrate_colors(samples)

    scores = {}

    # --------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------

    for i, disease in enumerate(disease_list):

        y = ROW_START[disease]

        sample = samples[i]

        risk = classify_bar(sample, centers, cluster_map)

        scores[disease] = risk

        if debug:

            color = COLOR_MAP.get(risk, (200,200,200))

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

    # --------------------------------------------------
    # DEBUG COLUMN LINES
    # --------------------------------------------------

    if debug:

        cv2.line(img, (X_LEFT,0), (X_LEFT,img.shape[0]), (255,0,0), 2)
        cv2.line(img, (X_RIGHT,0), (X_RIGHT,img.shape[0]), (255,0,0), 2)

        ok, png = cv2.imencode(".png", img)

        return png.tobytes()

    return {
        "engine": ENGINE_NAME,
        "scores": scores
    }
