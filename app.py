from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import gc

# ============================================================
# ITHRIVE HSV PRODUCTION ENGINE
# 150 DPI — HARD LOCKED COORDINATES (VISUALLY VERIFIED)
# Deterministic — No Layout Inference — No Fallback
# ============================================================

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION"
ENGINE_VERSION = "3.0.0_final_locked"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150

# ============================================================
# VERIFIED X WINDOW (INSIDE COLOR BARS)
# ============================================================

X_LEFT = 704
X_RIGHT = 710

# ============================================================
# VERIFIED Y COORDINATES (150 DPI, 20px HEIGHT)
# ============================================================

DISEASE_COORDINATES = {

    # PANEL 1
    "large_artery_stiffness": (689, 709),
    "peripheral_vessel": (714, 734),
    "blood_pressure_uncontrolled": (739, 759),
    "small_medium_artery_stiffness": (764, 784),
    "atherosclerosis": (789, 809),
    "ldl_cholesterol": (814, 834),
    "lv_hypertrophy": (839, 859),
    "metabolic_syndrome": (874, 894),
    "insulin_resistance": (899, 919),
    "beta_cell_function_decreased": (924, 944),
    "blood_glucose_uncontrolled": (949, 969),
    "tissue_inflammatory_process": (974, 994),

    # PANEL 2
    "hypothyroidism": (1145, 1165),
    "hyperthyroidism": (1170, 1190),
    "hepatic_fibrosis": (1195, 1215),
    "chronic_hepatitis": (1215, 1235),
    "prostate_cancer": (1235, 1255),
    "respiratory_disorders": (1260, 1280),
    "kidney_function_disorders": (1285, 1305),
    "digestive_disorders": (1310, 1330),
    "major_depression": (1355, 1375),
    "adhd_children_learning": (1380, 1400),
    "cerebral_dopamine_decreased": (1405, 1425),
    "cerebral_serotonin_decreased": (1425, 1445),
}

# ============================================================
# HSV CLASSIFICATION (ROBUST COLOR DETECTION)
# ============================================================

def classify_risk(roi):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    # Red (Severe)
    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = red_mask1 + red_mask2

    # Orange (Moderate)
    orange_mask = cv2.inRange(hsv, (15, 100, 100), (30, 255, 255))

    # Yellow (Mild)
    yellow_mask = cv2.inRange(hsv, (31, 100, 100), (65, 255, 255))

    red_pct = np.count_nonzero(red_mask) / total_pixels
    orange_pct = np.count_nonzero(orange_mask) / total_pixels
    yellow_pct = np.count_nonzero(yellow_mask) / total_pixels

    # Require meaningful saturation presence
    if max(red_pct, orange_pct, yellow_pct) < 0.05:
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
        "dpi": RENDER_DPI,
        "x_window": [X_LEFT, X_RIGHT]
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
            return jsonify({"error": "Missing page 2"}), 400

        page_image = np.array(pages[0])
        del pages
        gc.collect()

    except Exception:
        return jsonify({"error": "PDF processing failed"}), 500

    results = {}
