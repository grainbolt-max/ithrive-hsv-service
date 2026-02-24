import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "v56_hard_locked_geometry_anchor_1022"
API_KEY = "ithrive_secure_2026_key"

# =========================
# FIXED PAGE DIMENSIONS
# =========================
PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200

# =========================
# FIXED TRACK GEOMETRY
# =========================
ROW_COUNT = 12
ROW_HEIGHT = 32

# Final corrected anchor (measured + math adjusted)
PAGE1_FIRST_ROW_Y = 1022
PAGE2_FIRST_ROW_Y = 1022

TRACK_X1 = int(PAGE_WIDTH * 0.50)
TRACK_X2 = int(PAGE_WIDTH * 0.95)

# =========================
# PAGE 1 ORDER (First 12)
# =========================
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

# =========================
# PAGE 2 ORDER (Second 12)
# =========================
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
    return "HSV Preprocess Service Running v56"


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

    results = {}

    # =========================
    # PAGE 1
    # =========================
    page1 = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

    for i in range(ROW_COUNT):
        y = PAGE1_FIRST_ROW_Y + i * ROW_HEIGHT
        roi = page1[y:y + ROW_HEIGHT, TRACK_X1:TRACK_X2]

        percent = measure_yellow_span(roi)

        results[PAGE1_KEYS[i]] = {
            "progression_percent": percent,
            "risk_label": risk_label(percent),
            "source": ENGINE_NAME
        }

    # =========================
    # PAGE 2
    # =========================
    page2 = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)

    for i in range(ROW_COUNT):
        y = PAGE2_FIRST_ROW_Y + i * ROW_HEIGHT
        roi = page2[y:y + ROW_HEIGHT, TRACK_X1:TRACK_X2]

        percent = measure_yellow_span(roi)

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
