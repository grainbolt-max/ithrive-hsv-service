import os
import io
import numpy as np
import cv2
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"
ENGINE_NAME = "hsv_v40_calibrated_yellow_span_locked"

# ---------- HSV Calibration ----------
YELLOW_HUE_MIN = 18
YELLOW_HUE_MAX = 38
SAT_MIN = 50          # lowered from 70
VAL_MIN = 110         # lowered from 120

GRAY_RGB_DELTA = 12

ROW_HEIGHT = 48
TRACK_HEIGHT = 22
LEFT_MARGIN = 520
RIGHT_MARGIN = 1150

DISEASE_ROWS = [
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
    "tissue_inflammatory_process",
]

# --------------------------------------------------------

def is_gray_pixel(r, g, b):
    return (
        abs(int(r) - int(g)) < GRAY_RGB_DELTA and
        abs(int(r) - int(b)) < GRAY_RGB_DELTA and
        abs(int(g) - int(b)) < GRAY_RGB_DELTA
    )

def isolate_yellow_span(track_img):
    hsv = cv2.cvtColor(track_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    yellow_mask = (
        (h >= YELLOW_HUE_MIN) &
        (h <= YELLOW_HUE_MAX) &
        (s >= SAT_MIN) &
        (v >= VAL_MIN)
    )

    b, g, r = cv2.split(track_img)
    gray_mask = np.vectorize(is_gray_pixel)(r, g, b)

    final_mask = yellow_mask & (~gray_mask)

    cols = final_mask.any(axis=0)
    if not np.any(cols):
        return 0

    span_pixels = np.sum(cols)
    total_pixels = final_mask.shape[1]

    return int((span_pixels / total_pixels) * 100)


def classify(percent):
    if percent >= 80:
        return "severe"
    if percent >= 50:
        return "moderate"
    if percent >= 25:
        return "mild"
    if percent > 0:
        return "normal"
    return "none"


@app.route("/")
def health():
    return "HSV Preprocess Service Running v40"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        pdf_bytes = request.files["file"].read()
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except:
        return jsonify({"error": "PDF conversion failed"}), 500

    results = {}

    for page in pages:
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        base_y = 520

        for i, disease in enumerate(DISEASE_ROWS):
            if disease in results:
                continue

            y_center = base_y + (i * ROW_HEIGHT)
            y1 = int(y_center - TRACK_HEIGHT // 2)
            y2 = int(y_center + TRACK_HEIGHT // 2)

            track = img[y1:y2, LEFT_MARGIN:RIGHT_MARGIN]

            percent = isolate_yellow_span(track)

            results[disease] = {
                "progression_percent": percent,
                "risk_label": classify(percent),
                "source": ENGINE_NAME,
            }

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": len(pages),
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
