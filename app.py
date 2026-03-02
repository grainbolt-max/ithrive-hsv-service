from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import json
import gc

# ============================================================
# ITHRIVE HSV ENGINE — CALIBRATION BUILD
# X BAND HARD LOCKED TO 905–915
# Page 2 Only
# 150 DPI
# Deterministic
# No Inference
# No Fallback
# ============================================================

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION"
ENGINE_VERSION = "1.7.0_x_locked_905"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150
PAGE_INDEX = 1

SAT_GATE = 0.35
VAL_GATE = 0.35

# 🔒 CALIBRATION LOCK
LOCKED_X_LEFT = 905
LOCKED_X_RIGHT = 915

PANEL_1_KEYS = [
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
]

PANEL_2_KEYS = [
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

# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION
    })

# ============================================================
# HUE CLASSIFICATION
# ============================================================

def classify_hue(hue):
    if hue < 15 or hue > 345:
        return "Severe"
    if 15 <= hue < 40:
        return "Moderate"
    if 40 <= hue < 75:
        return "Mild"
    return "None/Low"

# ============================================================
# DETECTION
# ============================================================

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    layout_json = request.form.get("layout_profile")
    if not layout_json:
        return jsonify({"error": "layout_mismatch"}), 400

    try:
        layout = json.loads(layout_json)
    except Exception:
        return jsonify({"error": "layout_mismatch"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = request.files["file"].read()

    # ============================================================
    # MEMORY SAFE PAGE 2 RENDER ONLY
    # ============================================================

    try:
        pages = convert_from_bytes(
            pdf_bytes,
            dpi=RENDER_DPI,
            first_page=2,
            last_page=2
        )

        if not pages:
            return jsonify({"error": "layout_mismatch"}), 400

        page_image = np.array(pages[0])
        del pages
        gc.collect()

    except Exception:
        return jsonify({"error": "layout_mismatch"}), 400

    image_height, image_width = page_image.shape[:2]

    # 🔒 FORCE X BAND
    x_left = LOCKED_X_LEFT
    x_right = LOCKED_X_RIGHT

    results = {}

    for panel_name, keys in [
        ("panel_1", PANEL_1_KEYS),
        ("panel_2", PANEL_2_KEYS),
    ]:
        rows = layout["panels"][panel_name]["rows"]

        for key in keys:
            y_top = int(rows[key]["y_top"])
            y_bottom = int(rows[key]["y_bottom"])

            roi = page_image[y_top:y_bottom, x_left:x_right]

            if roi.size == 0:
                results[key] = "None/Low"
                continue

            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

            h = hsv[:, :, 0].astype(np.float32) * 2.0
            s = hsv[:, :, 1] / 255.0
            v = hsv[:, :, 2] / 255.0

            mask = (s > SAT_GATE) & (v > VAL_GATE)

            if not np.any(mask):
                results[key] = "None/Low"
            else:
                hue = float(np.median(h[mask]))
                results[key] = classify_hue(hue)

            del roi
            del hsv

    del page_image
    gc.collect()

    return jsonify({
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "dpi": RENDER_DPI,
        "x_left_used": x_left,
        "x_right_used": x_right,
        "results": results
    })

# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
