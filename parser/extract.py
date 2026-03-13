import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v68_blue_normalized_auto_calibrated"

# Locked sampling column
X_LEFT = 938
X_RIGHT = 955

# Detection window
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
# COLOR NORMALIZATION (convert risk colors → blue palette)
# ------------------------------------------------
def normalize_colors(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # yellow → cyan
    mask_yellow = cv2.inRange(hsv, (20,80,80), (35,255,255))
    hsv[:,:,0][mask_yellow > 0] = 90

    # orange → blue
    mask_orange = cv2.inRange(hsv, (10,80,80), (20,255,255))
    hsv[:,:,0][mask_orange > 0] = 110

    # red → dark blue
    mask_red1 = cv2.inRange(hsv, (0,80,80), (10,255,255))
    mask_red2 = cv2.inRange(hsv, (170,80,80), (180,255,255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    hsv[:,:,0][mask_red > 0] = 130

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


# ------------------------------------------------
# ROW DETECTION
# ------------------------------------------------
def detect_rows(img):

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
def sample_bar_color(img, y1, y2):

    mid = int((y1+y2)/2)

    square_left = X_LEFT + 2
    square_right = X_LEFT + 9

    sample = img[mid-2:mid+2, square_left:square_right]

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
def draw_debug(img, rows, scores):

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

    # NEW STAGE: color normalization
    img = normalize_colors(img)

    rows = detect_rows(img)

    samples = []

    for y1,y2 in rows:
        samples.append(sample_bar_color(img,y1,y2))

    centers,cluster_map = calibrate_colors(samples)

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

        overlay = draw_debug(img,rows,scores)

        _,png = cv2.imencode(".png",overlay)

        return png.tobytes()

    return {
        "engine":ENGINE_NAME,
        "scores":scores
    }


def extract_scores(pdf_bytes):

    return parse_report(pdf_bytes)
