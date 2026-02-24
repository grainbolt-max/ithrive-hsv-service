import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "hsv_v52_pure_coordinate_deterministic_locked"
API_KEY = "ithrive_secure_2026_key"

# ==== LOCKED GEOMETRY (Measured From 1700x2200 Page) ====
PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200

PANEL_X1 = 850
PANEL_X2 = 1630

PANEL_Y1 = 980
PANEL_Y2 = 1750

ROW_COUNT = 24
ROW_HEIGHT = int((PANEL_Y2 - PANEL_Y1) / ROW_COUNT)

DISEASE_KEYS = [
    "adhd_children_learning",
    "atherosclerosis",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "blood_pressure_uncontrolled",
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased",
    "chronic_hepatitis",
    "digestive_disorders",
    "hepatic_fibrosis",
    "hyperthyroidism",
    "hypothyroidism",
    "insulin_resistance",
    "kidney_function_disorders",
    "large_artery_stiffness",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "major_depression",
    "metabolic_syndrome",
    "peripheral_vessel",
    "prostate_cancer",
    "respiratory_disorders",
    "small_medium_artery_stiffness",
    "tissue_inflammatory_process"
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


def measure_yellow_span(roi):
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
    return "HSV Preprocess Service Running v52"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception:
        return jsonify({"error": "PDF conversion failed"}), 500

    # Disease bars live on page index 1
    page = pages[1]
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape

    if h != PAGE_HEIGHT or w != PAGE_WIDTH:
        return jsonify({
            "error": f"Unexpected page size {w}x{h}, expected 1700x2200"
        }), 400

    results = {}

    for i in range(ROW_COUNT):

        y1 = PANEL_Y1 + i * ROW_HEIGHT
        y2 = y1 + ROW_HEIGHT

        roi = image[y1:y2, PANEL_X1:PANEL_X2]

        percent = measure_yellow_span(roi)

        results[DISEASE_KEYS[i]] = {
            "progression_percent": percent,
            "risk_label": risk_label(percent),
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": len(pages),
        "row_height": ROW_HEIGHT,
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
