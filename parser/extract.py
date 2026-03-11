import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v68_auto_bar_band_detector"

# Detection window for disease table
MIN_Y = 880
MAX_Y = 2050

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
# DETECT BAR BAND (RIGHT → LEFT)
# ------------------------------------------------
def detect_bar_column(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]

    search_left = int(w * 0.60)
    search_right = int(w * 0.98)

    saturation_profile = []

    for x in range(search_left, search_right):

        column = hsv[MIN_Y:MAX_Y, x:x+3]

        s = column[:,:,1]

        score = np.sum(s > 40)

        saturation_profile.append(score)

    saturation_profile = np.array(saturation_profile)

    threshold = np.max(saturation_profile) * 0.4

    indices = np.where(saturation_profile > threshold)[0]

    if len(indices) == 0:
        return None, None

    left_edge = search_left + indices[0]
    right_edge = search_left + indices[-1]

    center = int((left_edge + right_edge) / 2)

    return center - 8, center + 8


# ------------------------------------------------
# ROW DETECTION
# ------------------------------------------------
def detect_rows(img, X_LEFT, X_RIGHT):

    column = img[MIN_Y:MAX_Y, X_LEFT:X_RIGHT]

    hsv = cv2.cvtColor(column, cv2.COLOR_BGR2HSV)

    rows = []
    inside = False
    start = 0

    for y in range(hsv.shape[0]):

        s = np.mean(hsv[y,:,1])

        if s > 40 and not inside:
            start = y
            inside = True

        if s < 20 and inside:

            end = y

            if 10 < (end-start) < 40:
                rows.append((start+MIN_Y,end+MIN_Y))

            inside = False

    filtered = []
    for r in rows:

        if not filtered:
            filtered.append(r)
            continue

        if r[0] - filtered[-1][0] > 18:
            filtered.append(r)

    return filtered


# ------------------------------------------------
# SAMPLE BAR COLOR
# ------------------------------------------------
def sample_bar_color(img, y1, y2, X_LEFT, X_RIGHT):

    mid = int((y1+y2)/2)

    sample = img[mid-2:mid+2, X_LEFT:X_RIGHT]

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:,:,0])
    s = np.mean(hsv[:,:,1])
    v = np.mean(hsv[:,:,2])

    return np.array([h,s,v])


# ------------------------------------------------
# CALIBRATE COLORS USING HUE
# ------------------------------------------------
def calibrate_colors(samples):

    colored = np.array([s for s in samples if s[1] > 25])

    if len(colored) < 3:
        return None

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
        order[0]:"yellow",
        order[1]:"orange",
        order[2]:"red"
    }

    return centers, cluster_map


# ------------------------------------------------
# CLASSIFY BAR
# ------------------------------------------------
def classify_bar(sample, centers, cluster_map):

    if sample[1] < 25:
        return None

    h = sample[0]

    dists = [abs(h - c) for c in centers]

    cluster = np.argmin(dists)

    return cluster_map[cluster]


# ------------------------------------------------
# DEBUG DRAW
# ------------------------------------------------
def draw_debug(img, rows, scores, X_LEFT, X_RIGHT):

    debug = img.copy()

    colors = {
        "yellow":(0,255,255),
        "orange":(0,165,255),
        "red":(0,0,255)
    }

    for i,(y1,y2) in enumerate(rows):

        if i >= len(DISEASES):
            break

        risk = scores.get(DISEASES[i])

        if risk is None:
            continue

        cv2.rectangle(
            debug,
            (X_LEFT,y1),
            (X_RIGHT,y2),
            colors[risk],
            3
        )

    return debug


# ------------------------------------------------
# MAIN PARSER
# ------------------------------------------------
def parse_report(pdf_bytes, debug=False):

    images = convert_from_bytes(pdf_bytes, dpi=200)

    img = np.array(images[1])

    X_LEFT, X_RIGHT = detect_bar_column(img)

    if X_LEFT is None:
        raise RuntimeError("Failed to detect disease bar column")

    rows = detect_rows(img, X_LEFT, X_RIGHT)

    samples = []

    for y1,y2 in rows:
        samples.append(sample_bar_color(img,y1,y2,X_LEFT,X_RIGHT))

    calibration = calibrate_colors(samples)

    if calibration is None:
        raise RuntimeError("Color calibration failed")

    centers,cluster_map = calibration

    scores = {}

    for i,(y1,y2) in enumerate(rows):

        if i >= len(DISEASES):
            break

        scores[DISEASES[i]] = classify_bar(
            samples[i],
            centers,
            cluster_map
        )

    if debug:

        overlay = draw_debug(img,rows,scores,X_LEFT,X_RIGHT)

        _,png = cv2.imencode(".png",overlay)

        return png.tobytes()

    return {
        "engine":ENGINE_NAME,
        "scores":scores
    }


# REQUIRED ENTRYPOINT FOR app.py
def extract_disease_scores(pdf_bytes, *args, **kwargs):
    return parse_report(pdf_bytes)
