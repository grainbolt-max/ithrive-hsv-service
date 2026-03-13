import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v70_width_based_classifier"

CURRENT_DPI = 200

# Relative column position (percentage of page width)
BAR_X_RATIO = 0.46
BAR_WIDTH_RATIO = 0.015

# Detection window ratios
MIN_Y_RATIO = 0.42
MAX_Y_RATIO = 0.98
ROW_HEIGHT_RATIO = 0.022

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
    "cerebral_serotonin_decreased",
]


def detect_rows(img_height):
    min_y = int(img_height * MIN_Y_RATIO)
    row_height = int(img_height * ROW_HEIGHT_RATIO)

    rows = []
    y = min_y

    for _ in range(len(DISEASES)):
        rows.append(int(y))
        y += row_height

    return rows


def compute_bar_column(img_width):
    x_left = int(img_width * BAR_X_RATIO)
    x_right = x_left + int(img_width * BAR_WIDTH_RATIO)
    return x_left, x_right


def sample_bar_color(img, y, x_left, x_right):
    crop = img[y - 6:y + 6, x_left:x_right]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])

    return h, s, v


def classify_bar(h, s, v):

    if s < 40:
        return None

    if h < 10:
        return "red"

    if h < 25:
        return "orange"

    if h < 45:
        return "yellow"

    return None


def draw_debug(img, rows, scores, x_left, x_right):
    overlay = img.copy()

    height = img.shape[0]

    cv2.rectangle(
        overlay,
        (x_left, int(height * MIN_Y_RATIO)),
        (x_right, int(height * MAX_Y_RATIO)),
        (255, 0, 0),
        2
    )

    for y in rows:
        cv2.rectangle(
            overlay,
            (x_left - 10, y - 8),
            (x_right + 10, y + 8),
            (0, 255, 0),
            1
        )

    return overlay


def parse_report(pdf_bytes, debug=False):

    pages = convert_from_bytes(pdf_bytes, dpi=CURRENT_DPI)

    if len(pages) < 2:
        raise Exception("PDF missing disease screening page")

    page = pages[1]

    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]

    x_left, x_right = compute_bar_column(width)

    rows = detect_rows(height)

    scores = {}

    for disease, y in zip(DISEASES, rows):

        color = sample_bar_color(img, y, x_left, x_right)

        if color is None:
            scores[disease] = None
            continue

        h, s, v = color

        scores[disease] = classify_bar(h, s, v)

    if debug:
        overlay = draw_debug(img, rows, scores, x_left, x_right)
        ok, png = cv2.imencode(".png", overlay)
        return png.tobytes()

    ordered_scores = {d: scores.get(d) for d in DISEASES}

    return {
        "engine": ENGINE_NAME,
        "scores": ordered_scores
    }


def extract_scores(pdf_bytes, debug=False):
    return parse_report(pdf_bytes, debug=debug)
