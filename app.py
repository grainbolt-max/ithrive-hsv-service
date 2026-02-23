# ==============================
# v31 STRICT SATURATION SPAN LOCKED
# ==============================

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "hsv_v31_strict_saturation_span_locked"
AUTH_KEY = "ithrive_secure_2026_key"

# --- CONFIGURATION ---

# Saturation threshold to remove gray background
SATURATION_THRESHOLD = 50

# Bar geometry (locked)
BAR_LEFT = 350
BAR_RIGHT = 1100
BAR_HEIGHT = 28

# Vertical spacing
ROW_START_Y = 420
ROW_GAP = 42

# 24 diseases in fixed order
DISEASE_ORDER = [
    # Page 1
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

    # Page 2
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

    # Convert safely to BGR if RGBA
    if roi.shape[2] == 4:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Strict saturation isolate
    s_channel = hsv[:, :, 1]
    color_mask = s_channel > SATURATION_THRESHOLD

    # Find horizontal extent
    cols = np.where(np.any(color_mask, axis=0))[0]

    if len(cols) == 0:
        return 0

    rightmost = cols[-1]
    percent = int((rightmost / roi.shape[1]) * 100)

    return percent


@app.route("/")
def home():
    return "HSV Preprocess Service Running v31"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {AUTH_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"].read()

    try:
        pages = convert_from_bytes(file, dpi=300)
    except:
        return jsonify({"error": "PDF conversion failed"}), 500

    results = {}
    disease_index = 0

    for page in pages[:2]:  # Only first 2 pages contain bars
        image = np.array(page)

        for _ in range(12):
            if disease_index >= len(DISEASE_ORDER):
                break

            percent = analyze_bar(image, disease_index % 12)
            label = risk_label_from_percent(percent)

            results[DISEASE_ORDER[disease_index]] = {
                "progression_percent": percent,
                "risk_label": label,
                "source": ENGINE_NAME
            }

            disease_index += 1

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": len(pages),
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
