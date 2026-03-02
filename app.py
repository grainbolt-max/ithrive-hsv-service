from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import json
import gc

# ============================================================
# PRODUCTION ENGINE (DEBUG WIDTH MODE)
# Stateless Layout-Driven HSV Disease Classifier
# LOW MEMORY MODE (Free Tier Safe)
# ============================================================

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION"
ENGINE_VERSION = "1.2.0_low_memory_DEBUG"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150   # current production DPI
PAGE_INDEX = 1     # Page 2 (0-based)

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

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION
    })

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

    # ============================================================
    # DEBUG RETURN — TEMPORARY
    # ============================================================

    return jsonify({
        "debug_width": image_width,
        "debug_height": image_height,
        "dpi": RENDER_DPI
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
