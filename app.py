from flask import Flask, request, jsonify
import os
import sys
import numpy as np
import cv2
from pdf2image import convert_from_bytes

# ============================================================
# FINAL STABLE PRODUCTION ENGINE
# ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION
# Version 1.3.0
# Deterministic. No inference. No dynamic measurement.
# ============================================================

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION"
ENGINE_VERSION = "1.3.0"

API_KEY = os.environ.get("ITHRIVE_API_KEY")

if not API_KEY:
    print("FATAL ERROR: ITHRIVE_API_KEY not set")
    sys.exit(1)

app = Flask(__name__)

# ============================================================
# FIXED RISK COLOR WINDOW (LOCKED)
# ============================================================

X_LEFT = 1412
X_RIGHT = 1422

# ============================================================
# FIXED COORDINATE MAP (PAGE 2 ONLY)
# ============================================================

DISEASE_COORDINATES = {

    # PANEL 1
    "large_artery_stiffness": (1375, 1400),
    "peripheral_vessel": (1425, 1450),
    "blood_pressure_uncontrolled": (1475, 1500),
    "small_medium_artery_stiffness": (1525, 1550),
    "atherosclerosis": (1575, 1600),
    "ldl_cholesterol": (1625, 1650),
    "lv_hypertrophy": (1675, 1700),
    "metabolic_syndrome": (1750, 1775),
    "insulin_resistance": (1800, 1825),
    "beta_cell_function_decreased": (1850, 1875),
    "blood_glucose_uncontrolled": (1900, 1925),
    "tissue_inflammatory_process": (1950, 1975),

    # PANEL 2
    "hypothyroidism": (2290, 2320),
    "hyperthyroidism": (2340, 2365),
    "hepatic_fibrosis": (2390, 2420),
    "chronic_hepatitis": (2430, 2455),
    "prostate_cancer": (2475, 2500),
    "respiratory_disorders": (2525, 2550),
    "kidney_function_disorders": (2575, 2600),
    "digestive_disorders": (2625, 2650),
    "major_depression": (2720, 2745),
    "adhd_children_learning": (2760, 2785),
    "cerebral_dopamine_decreased": (2815, 2840),
    "cerebral_serotonin_decreased": (2850, 2875),
}

# ============================================================
# ORIGINAL PROVEN MASK-BASED HSV CLASSIFIER
# ============================================================

def classify_risk(bgr_roi):

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = red_mask1 + red_mask2

    orange_mask = cv2.inRange(hsv, (15, 100, 100), (30, 255, 255))
    yellow_mask = cv2.inRange(hsv, (31, 100, 100), (65, 255, 255))

    red_pct = np.count_nonzero(red_mask) / total_pixels
    orange_pct = np.count_nonzero(orange_mask) / total_pixels
    yellow_pct = np.count_nonzero(yellow_mask) / total_pixels

    # No strong color
    if red_pct < 0.05 and orange_pct < 0.05 and yellow_pct < 0.05:
        return "None/Low"

    # Dominant color
    if red_pct >= orange_pct and red_pct >= yellow_pct:
        return "Severe"

    if orange_pct >= red_pct and orange_pct >= yellow_pct:
        return "Moderate"

    if yellow_pct >= red_pct and yellow_pct >= orange_pct:
        return "Mild"

    return "None/Low"


# ============================================================
# DETERMINISTIC LAYOUT MISMATCH GATE
# ============================================================

def layout_alignment_valid(page_image):
    """
    Deterministic structural sanity check.

    1. Background immediately left of X_LEFT must be neutral gray.
    2. First disease row color window must contain color (non-gray).
    """

    # --- Check background left of color window ---
    bg_roi = page_image[1375:1400, X_LEFT-40:X_LEFT-20]

    mean_b = np.mean(bg_roi[:, :, 0])
    mean_g = np.mean(bg_roi[:, :, 1])
    mean_r = np.mean(bg_roi[:, :, 2])

    if abs(mean_r - mean_g) > 15 or abs(mean_g - mean_b) > 15:
        return False

    # --- Check first row contains color ---
    first_row = page_image[1375:1400, X_LEFT:X_RIGHT]
    hsv = cv2.cvtColor(first_row, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] / 255.0

    if np.mean(sat) < 0.02:
        return False

    return True


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION
    })


# ============================================================
# MAIN DETECTION ENDPOINT
# ============================================================

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = request.files["file"].read()
    pages = convert_from_bytes(pdf_bytes, dpi=300)

    if len(pages) < 2:
        return jsonify({"error": "PDF must have at least 2 pages"}), 400

    page_image = np.array(pages[1])
    page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)

    # ========================================================
    # Layout mismatch gate
    # ========================================================

    if not layout_alignment_valid(page_image):
        return jsonify({
            "error": "layout_mismatch",
            "engine": ENGINE_NAME,
            "version": ENGINE_VERSION
        }), 400

    # ========================================================
    # Classification
    # ========================================================

    results = {}

    for disease, (y_top, y_bottom) in DISEASE_COORDINATES.items():
        roi = page_image[y_top:y_bottom, X_LEFT:X_RIGHT]
        severity = classify_risk(roi)
        results[disease] = severity

    return jsonify({
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "page_measured": 2,
        "x_window": [X_LEFT, X_RIGHT],
        "results": results
    })


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
