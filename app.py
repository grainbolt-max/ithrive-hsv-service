from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import json
from pdf2image import convert_from_bytes

# ============================================================
# PRODUCTION ENGINE
# Stateless Layout-Driven HSV Disease Classifier
# Deterministic — No Inference — No Fallback
# Renderer: pdf2image (calibrated baseline)
# ============================================================

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION"
ENGINE_VERSION = "1.1.0"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 300
PAGE_INDEX = 1

SAT_GATE = 0.35
VAL_GATE = 0.35

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
# HEALTH
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
# STRICT LAYOUT VALIDATION
# ============================================================

def validate_layout(layout, image_width, image_height):
    x_left = layout.get("x_left")
    x_right = layout.get("x_right")
    panels = layout.get("panels", {})

    if x_left is None or x_right is None:
        return False

    width = x_right - x_left
    if width < 5 or width > 50:
        return False

    if x_right > image_width:
        return False

    panel_1 = panels.get("panel_1", {}).get("rows", {})
    panel_2 = panels.get("panel_2", {}).get("rows", {})

    for key in PANEL_1_KEYS:
        if key not in panel_1:
            return False

    for key in PANEL_2_KEYS:
        if key not in panel_2:
            return False

    for keys, panel in [
        (PANEL_1_KEYS, panel_1),
        (PANEL_2_KEYS, panel_2),
    ]:
        prev_bottom = -1
        for key in keys:
            y_top = int(panel[key]["y_top"])
            y_bottom = int(panel[key]["y_bottom"])

            if y_top >= y_bottom:
                return False

            if y_top < prev_bottom:
                return False

            if y_bottom > image_height:
                return False

            prev_bottom = y_bottom

    return True

# ============================================================
# DETECTION
# ============================================================

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
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

    # ---- RENDER USING pdf2image (calibrated baseline) ----
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=RENDER_DPI)

        if len(pages) <= PAGE_INDEX:
            return jsonify({"error": "layout_mismatch"}), 400

        page_image = np.array(pages[PAGE_INDEX])
        img = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)

    except Exception:
        return jsonify({"error": "layout_mismatch"}), 400

    image_height, image_width = img.shape[:2]

    valid = validate_layout(layout, image_width, image_height)
    if not valid:
        return jsonify({
            "error": "layout_mismatch"
        }), 400

    x_left = layout["x_left"]
    x_right = layout["x_right"]

    panel_1 = layout["panels"]["panel_1"]["rows"]
    panel_2 = layout["panels"]["panel_2"]["rows"]

    results = {}

    for panel_keys, panel_rows in [
        (PANEL_1_KEYS, panel_1),
        (PANEL_2_KEYS, panel_2),
    ]:
        for key in panel_keys:
            y_top = int(panel_rows[key]["y_top"])
            y_bottom = int(panel_rows[key]["y_bottom"])

            row_img = img[y_top:y_bottom, x_left:x_right]

            if row_img.size == 0:
                results[key] = "None/Low"
                continue

            hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)

            h = hsv[:, :, 0].astype(np.float32) * 2.0
            s = hsv[:, :, 1] / 255.0
            v = hsv[:, :, 2] / 255.0

            mask = (s > SAT_GATE) & (v > VAL_GATE)

            if not np.any(mask):
                results[key] = "None/Low"
                continue

            hue = float(np.median(h[mask]))
            results[key] = classify_hue(hue)

    return jsonify({
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
