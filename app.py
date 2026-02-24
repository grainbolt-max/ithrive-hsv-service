import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "v53_anchor_locked_deterministic"
API_KEY = "ithrive_secure_2026_key"

ROW_COUNT = 12
ROW_HEIGHT = 32

PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200

# PAGE 1 diseases (first 12)
PAGE1_KEYS = [
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
    "tissue_inflammatory_process"
]

# PAGE 2 diseases (second 12)
PAGE2_KEYS = [
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


def risk_label(percent):
    if percent >= 75:
        return "severe"
    elif percent >= 50:
        return "moderate"
    elif percent >= 20:
        return "mild"
    elif percent > 0:
        return "normal"
    else:
        return "none"


def find_first_track_y(image):
    """
    Detect first horizontal gray track row using horizontal projection.
    """

    h, w, _ = image.shape

    # right side panel only
    panel = image[int(h * 0.40):int(h * 0.85),
                  int(w * 0.50):int(w * 0.95)]

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    # horizontal mean intensity
    row_means = np.mean(gray, axis=1)

    # tracks are darker than background
    threshold = np.mean(row_means) - 15

    for i, val in enumerate(row_means):
        if val < threshold:
            return int(h * 0.40) + i

    return int(h * 0.40)


def measure_row(image, y_start):
    y1 = y_start
    y2 = y1 + ROW_HEIGHT

    x1 = int(PAGE_WIDTH * 0.50)
    x2 = int(PAGE_WIDTH * 0.95)

    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 110, 120])
    upper_yellow = np.array([38, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0

    span = xs.max() - xs.min()
    percent = int((span / roi.shape[1]) * 100)

    if percent < 10:
        return 0

    return min(percent, 100)


@app.route("/")
def home():
    return "HSV Preprocess Service Running v53"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    file = request.files["file"]
    pdf_bytes = file.read()

    pages = convert_from_bytes(pdf_bytes, dpi=200)

    results = {}

    # -------- PAGE 1 --------
    page1 = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    first_y1 = find_first_track_y(page1)

    for i in range(ROW_COUNT):
        y = first_y1 + i * ROW_HEIGHT
        percent = measure_row(page1, y)

        results[PAGE1_KEYS[i]] = {
            "progression_percent": percent,
            "risk_label": risk_label(percent),
            "source": ENGINE_NAME
        }

    # -------- PAGE 2 --------
    page2 = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)
    first_y2 = find_first_track_y(page2)

    for i in range(ROW_COUNT):
        y = first_y2 + i * ROW_HEIGHT
        percent = measure_row(page2, y)

        results[PAGE2_KEYS[i]] = {
            "progression_percent": percent,
            "risk_label": risk_label(percent),
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": len(pages),
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
