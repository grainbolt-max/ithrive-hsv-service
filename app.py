import os
import io
import base64
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")

ENGINE_NAME = "hsv_v17_locked"

# ==============================
# PAGE ROW DEFINITIONS (LOCKED)
# ==============================

PAGE_1_KEYS = [
    "large_artery_stiffness",
    "peripheral_vessels",
    "blood_glucose_uncontrolled",
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

PAGE_2_KEYS = [
    "hypothyroidism",
    "hyperthyroidism",
    "hepatic_fibrosis",
    "chronic_hepatitis",
    "respiratory",
    "kidney_function",
    "digestive_disorders",
    "major_depression",
    "adhd_children_learning",
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased"
]

ALL_KEYS = list(set(PAGE_1_KEYS + PAGE_2_KEYS))

# ==============================
# HEALTH
# ==============================

@app.route("/", methods=["GET"])
def health():
    return f"HSV Preprocess Service Running v17", 200

# ==============================
# CORE BAR DETECTION
# ==============================

def detect_bars_from_page(image, expected_keys):
    h, w, _ = image.shape

    # Crop Disease Score column (locked by screenshot analysis)
    x1 = int(w * 0.48)
    x2 = int(w * 0.85)
    y1 = int(h * 0.18)
    y2 = int(h * 0.88)

    score_col = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(score_col, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Detect colored rows by saturation projection
    row_strength = np.mean(saturation, axis=1)

    threshold = 25
    colored_rows = row_strength > threshold

    segments = []
    start = None

    for i, val in enumerate(colored_rows):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start > 8:
                segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(colored_rows)))

    results = {}

    segments = sorted(segments, key=lambda x: x[0])

    for idx, seg in enumerate(segments):
        if idx >= len(expected_keys):
            break

        key = expected_keys[idx]

        y_top = seg[0]
        y_bot = seg[1]

        bar_img = hsv[y_top:y_bot, :, :]

        hue = bar_img[:, :, 0]
        sat = bar_img[:, :, 1]

        mask = sat > 30
        if np.sum(mask) == 0:
            continue

        avg_hue = np.mean(hue[mask])

        if 20 <= avg_hue <= 35:
            label = "mild"
            percent = 50
        elif 0 <= avg_hue <= 10:
            label = "severe"
            percent = 85
        elif 45 <= avg_hue <= 80:
            label = "normal"
            percent = 10
        else:
            label = "none"
            percent = 20

        results[key] = {
            "progression_percent": percent,
            "risk_label": label,
            "source": ENGINE_NAME
        }

    return results

# ==============================
# DETECT ENDPOINT
# ==============================

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():
    try:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        pdf_bytes = file.read()

        pages = convert_from_bytes(pdf_bytes, dpi=200)

        results = {}

        if len(pages) >= 1:
            img1 = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            results.update(detect_bars_from_page(img1, PAGE_1_KEYS))

        if len(pages) >= 2:
            img2 = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)
            results.update(detect_bars_from_page(img2, PAGE_2_KEYS))

        # Ensure all expected keys exist
        for key in ALL_KEYS:
            if key not in results:
                results[key] = {
                    "progression_percent": 20,
                    "risk_label": "none",
                    "source": ENGINE_NAME
                }

        return jsonify({
            "engine": ENGINE_NAME,
            "pages_found": len(pages),
            "results": results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
