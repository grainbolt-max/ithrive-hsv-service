# ===================================================
# v37 STRICT HUE + SATURATION SPAN (FIXED GEOMETRY)
# Page-aware geometry
# PyMuPDF @ 300 DPI
# Deterministic. No track detection.
# ===================================================

import fitz
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

ENGINE_NAME = "hsv_v37_fixed_span_strict_locked"
AUTH_KEY = "ithrive_secure_2026_key"

# ----------------------------
# CONFIG
# ----------------------------

SATURATION_THRESHOLD = 30
MIN_COLUMN_DENSITY = 0.4

BAR_LEFT = 350
BAR_RIGHT = 1100
BAR_HEIGHT = 26
ROW_GAP = 44

PAGE1_ROW_START_Y = 455
PAGE2_ROW_START_Y = 430

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

# ----------------------------
# RISK LABEL
# ----------------------------

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

# ----------------------------
# COLOR MASK
# ----------------------------

def disease_color_mask(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    red_mask = ((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 180))
    orange_mask = (h > 10) & (h <= 20)
    yellow_mask = (h > 20) & (h <= 35)

    return (s > SATURATION_THRESHOLD) & (red_mask | orange_mask | yellow_mask)

# ----------------------------
# ANALYZE BAR
# ----------------------------

def analyze_bar(image, base_y, row_index):
    y1 = base_y + (row_index * ROW_GAP)
    y2 = y1 + BAR_HEIGHT

    roi = image[y1:y2, BAR_LEFT:BAR_RIGHT]

    if roi.size == 0:
        return 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = disease_color_mask(hsv)

    col_density = np.sum(mask, axis=0) / mask.shape[0]
    valid_cols = np.where(col_density > MIN_COLUMN_DENSITY)[0]

    if len(valid_cols) == 0:
        return 0

    fill_end = valid_cols[-1]
    total_width = roi.shape[1]

    percent = int((fill_end / total_width) * 100)
    percent = max(0, min(percent, 100))

    return percent

# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def home():
    return "HSV Preprocess Service Running v37"

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {AUTH_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file_bytes = request.files["file"].read()

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except:
        return jsonify({"error": "PDF open failed"}), 500

    results = {}
    disease_index = 0

    for page_number in range(min(2, len(doc))):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=300)

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        base_y = PAGE1_ROW_START_Y if page_number == 0 else PAGE2_ROW_START_Y

        for row in range(12):
            if disease_index >= len(DISEASE_ORDER):
                break

            percent = analyze_bar(img, base_y, row)
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
