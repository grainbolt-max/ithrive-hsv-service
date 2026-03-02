from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import gc

# ============================================================
# ITHRIVE PRODUCTION ENGINE
# 150 DPI HARD-LOCKED COORDINATE VERSION
# Deterministic — No Layout Inference — No Fallback
# ============================================================

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION"
ENGINE_VERSION = "2.0.0_150dpi_locked"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150
PAGE_INDEX = 1

# ============================================================
# LOCKED X WINDOW (Verified Visually)
# ============================================================

X_LEFT = 703
X_RIGHT = 708

# ============================================================
# LOCKED Y COORDINATES (150 DPI SCALED)
# ============================================================

DISEASE_COORDINATES = {

    # PANEL 1
    "large_artery_stiffness": (688, 700),
    "peripheral_vessel": (713, 725),
    "blood_pressure_uncontrolled": (738, 750),
    "small_medium_artery_stiffness": (763, 775),
    "atherosclerosis": (788, 800),
    "ldl_cholesterol": (813, 825),
    "lv_hypertrophy": (838, 850),
    "metabolic_syndrome": (875, 888),
    "insulin_resistance": (900, 913),
    "beta_cell_function_decreased": (925, 938),
    "blood_glucose_uncontrolled": (950, 963),
    "tissue_inflammatory_process": (975, 988),

    # PANEL 2
    "hypothyroidism": (1145, 1160),
    "hyperthyroidism": (1170, 1183),
    "hepatic_fibrosis": (1195, 1210),
    "chronic_hepatitis": (1215, 1228),
    "prostate_cancer": (1238, 1250),
    "respiratory_disorders": (1263, 1275),
    "kidney_function_disorders": (1288, 1300),
    "digestive_disorders": (1313, 1325),
    "major_depression": (1360, 1373),
    "adhd_children_learning": (1380, 1393),
    "cerebral_dopamine_decreased": (1408, 1420),
    "cerebral_serotonin_decreased": (1425, 1438),
}

# ============================================================
# STRICT HSV COLOR CLASSIFICATION
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

    if red_pct < 0.05 and orange_pct < 0.05 and yellow_pct < 0.05:
        return "None/Low"

    if red_pct >= orange_pct and red_pct >= yellow_pct:
        return "Severe"

    if orange_pct >= red_pct and orange_pct >= yellow_pct:
        return "Moderate"

    if yellow_pct >= red_pct and yellow_pct >= orange_pct:
        return "Mild"

    return "None/Low"

# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "dpi": RENDER_DPI
    })

# ============================================================
# DETECTION ENDPOINT
# ============================================================

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():

    if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = request.files["file"].read()

    try:
        pages = convert_from_bytes(
            pdf_bytes,
            dpi=RENDER_DPI,
            first_page=2,
            last_page=2
        )

        if not pages:
            return jsonify({"error": "PDF missing page 2"}), 400

        page_image = np.array(pages[0])
        del pages
        gc.collect()

    except Exception:
        return jsonify({"error": "PDF processing failed"}), 500

    results = {}

    for disease, (y_top, y_bottom) in DISEASE_COORDINATES.items():

        roi = page_image[y_top:y_bottom, X_LEFT:X_RIGHT]

        if roi.size == 0:
            results[disease] = "None/Low"
            continue

        severity = classify_risk(roi)
        results[disease] = severity

    del page_image
    gc.collect()

    return jsonify({
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "dpi": RENDER_DPI,
        "x_window": [X_LEFT, X_RIGHT],
        "results": results
    })


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
