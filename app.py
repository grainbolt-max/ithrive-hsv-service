import base64
import io
import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

ENGINE_VERSION = "hsv_v30_strict_isolate_locked"
AUTH_KEY = "ithrive_secure_2026_key"

# ================================
# Disease Order (Exact Row Order)
# ================================

PAGE_1 = [
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
    "tissue_inflammatory_process"
]

PAGE_2 = [
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
    "cerebral_serotonin_decreased"
]

ALL_DISEASES = PAGE_1 + PAGE_2

# =====================================
# HSV STRICT COLOR ISOLATION
# =====================================

def positive_color_mask(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # -------------------------
    # Exclude dark gray background
    # Dark gray has LOW brightness
    # -------------------------
    bright = v > 120

    # -------------------------
    # Yellow
    # -------------------------
    yellow = (
        (h >= 18) & (h <= 35) &
        (s > 80) &
        bright
    )

    # -------------------------
    # Orange
    # -------------------------
    orange = (
        (h >= 8) & (h < 18) &
        (s > 100) &
        bright
    )

    # -------------------------
    # Red
    # -------------------------
    red = (
        ((h <= 6) | (h >= 170)) &
        (s > 120) &
        bright
    )

    # -------------------------
    # Light Neutral Gray (None/Low)
    # Low saturation, HIGH brightness
    # -------------------------
    neutral_light = (
        (s < 40) &
        (v > 170)
    )

    mask = yellow | orange | red | neutral_light
    return mask.astype(np.uint8) * 255


# =====================================
# Span Detection
# =====================================

def detect_span_percent(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = positive_color_mask(hsv)

    height, width = mask.shape

    # Collapse vertically (if ANY pixel in column is positive)
    col_has_color = np.any(mask > 0, axis=0)

    # Find last colored column from left
    colored_indices = np.where(col_has_color)[0]

    if len(colored_indices) == 0:
        return 0

    last_index = colored_indices[-1]
    percent = int((last_index / width) * 100)

    return percent


# =====================================
# Risk Label
# =====================================

def risk_from_percent(p):
    if p >= 75:
        return "severe"
    elif p >= 50:
        return "moderate"
    elif p >= 25:
        return "mild"
    elif p > 0:
        return "none"
    else:
        return "normal"


# =====================================
# PDF Processing
# =====================================

def process_pdf(file_stream):
    results = {}

    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    pages_found = len(doc)

    # Hard geometry crop region for bars
    # Adjust if needed once stable
    X_START = 900
    X_END = 1700

    Y_START_PAGE1 = 700
    ROW_HEIGHT = 90
    ROW_SPACING = 95

    # PAGE 1
    page1 = doc[0]
    pix1 = page1.get_pixmap()
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8)
    img1 = img1.reshape(pix1.height, pix1.width, pix1.n)

    for i, disease in enumerate(PAGE_1):
        y1 = Y_START_PAGE1 + i * ROW_SPACING
        y2 = y1 + ROW_HEIGHT

        crop = img1[y1:y2, X_START:X_END]

        percent = detect_span_percent(crop)
        label = risk_from_percent(percent)

        results[disease] = {
            "progression_percent": percent,
            "risk_label": label,
            "source": ENGINE_VERSION
        }

    # PAGE 2
    page2 = doc[1]
    pix2 = page2.get_pixmap()
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8)
    img2 = img2.reshape(pix2.height, pix2.width, pix2.n)

    Y_START_PAGE2 = 700

    for i, disease in enumerate(PAGE_2):
        y1 = Y_START_PAGE2 + i * ROW_SPACING
        y2 = y1 + ROW_HEIGHT

        crop = img2[y1:y2, X_START:X_END]

        percent = detect_span_percent(crop)
        label = risk_from_percent(percent)

        results[disease] = {
            "progression_percent": percent,
            "risk_label": label,
            "source": ENGINE_VERSION
        }

    return results, pages_found


# =====================================
# Routes
# =====================================

@app.route("/", methods=["GET"])
def root():
    return f"HSV Preprocess Service Running v30"

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {AUTH_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    results, pages_found = process_pdf(file)

    return jsonify({
        "engine": ENGINE_VERSION,
        "pages_found": pages_found,
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
