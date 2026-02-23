# ==============================
# v31.1 STRICT SATURATION SPAN LOCKED (PyMuPDF)
# ==============================

import fitz  # PyMuPDF
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

ENGINE_NAME = "hsv_v31_1_strict_saturation_span_locked"
AUTH_KEY = "ithrive_secure_2026_key"

SATURATION_THRESHOLD = 50

BAR_LEFT = 350
BAR_RIGHT = 1100
BAR_HEIGHT = 28

ROW_START_Y = 420
ROW_GAP = 42

DISEASE_ORDER = [
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


def risk_label_from_percent(p):
    if p >= 75:
        return "severe"
    elif p >= 50:
        return "moderate"
    elif p >= 25:
        return "mild"
    elif p >= 10:
        return "normal"
    else:
        return "none"


def analyze_bar(image, row_index):
    y1 = ROW_START_Y + (row_index * ROW_GAP)
    y2 = y1 + BAR_HEIGHT

    roi = image[y1:y2, BAR_LEFT:BAR_RIGHT]

    if roi.size == 0:
        return 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    s_channel = hsv[:, :, 1]
    mask = s_channel > SATURATION_THRESHOLD

    cols = np.where(np.any(mask, axis=0))[0]

    if len(cols) == 0:
        return 0

    rightmost = cols[-1]
    percent = int((rightmost / roi.shape[1]) * 100)

    return percent


@app.route("/")
def home():
    return "HSV Preprocess Service Running v31.1"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {AUTH_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"].read()

    try:
        doc = fitz.open(stream=file, filetype="pdf")
    except:
        return jsonify({"error": "PDF open failed"}), 500

    results = {}
    disease_index = 0

    for page_number in range(min(2, len(doc))):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        for row in range(12):
            if disease_index >= len(DISEASE_ORDER):
                break

            percent = analyze_bar(img, row)
            label = risk_label_from_percent(percent)

            results[DISEASE_ORDER[disease_index]] = {
                "progression_percent": percent,
                "risk_label": label,
                "source": ENGINE_NAME
            }

            disease_index += 1

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": len(doc),
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
