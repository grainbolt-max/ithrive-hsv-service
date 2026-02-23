import os
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")
ENGINE_NAME = "hsv_v25_geometry_center_locked"


# =========================================================
# FIXED DISEASE ORDER (STRICT TEMPLATE MATCH)
# =========================================================

PAGE_1_KEYS = [
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

PAGE_2_KEYS = [
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

ALL_KEYS = set(PAGE_1_KEYS) | set(PAGE_2_KEYS)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running v25", 200


# =========================================================
# FIXED GEOMETRY + CENTER SAMPLING
# =========================================================

def extract_rows_geometry_center_locked(image, disease_keys):
    h, w, _ = image.shape

    # Fixed score column crop (template locked)
    x1 = int(w * 0.48)
    x2 = int(w * 0.85)
    y1 = int(h * 0.20)
    y2 = int(h * 0.88)

    col = image[y1:y2, x1:x2]
    col_h, col_w = col.shape[:2]

    total_rows = 14  # 1 heading + 12 diseases + 1 section gap
    row_height = col_h // total_rows

    results = {}

    for i, key in enumerate(disease_keys):
        row_index = i + 1  # skip heading row (row 0)

        top = row_index * row_height
        bottom = (row_index + 1) * row_height

        row_crop = col[top:bottom, :]

        # CENTER SAMPLE WINDOW (deterministic)
        rh, rw = row_crop.shape[:2]

        cy1 = int(rh * 0.25)
        cy2 = int(rh * 0.75)

        cx1 = int(rw * 0.30)
        cx2 = int(rw * 0.70)

        center_crop = row_crop[cy1:cy2, cx1:cx2]

        label, percent = classify_color(center_crop)

        results[key] = {
            "progression_percent": percent,
            "risk_label": label,
            "source": ENGINE_NAME
        }

    return results


# =========================================================
# COLOR CLASSIFICATION (STRICT SATURATION FILTER)
# =========================================================

def classify_color(bgr_crop):
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    # Remove dark gray background
    mask = sat > 40

    if np.sum(mask) == 0:
        return "none", 20

    avg_hue = np.mean(hue[mask])

    if 0 <= avg_hue <= 10:
        return "severe", 85
    elif 10 < avg_hue <= 20:
        return "moderate", 70
    elif 20 < avg_hue <= 35:
        return "mild", 50
    elif 45 <= avg_hue <= 80:
        return "normal", 10
    else:
        return "none", 20


# =========================================================
# MAIN ENDPOINT
# =========================================================

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    try:
        if request.headers.get("Authorization") != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        pdf_bytes = request.files["file"].read()
        pages = convert_from_bytes(pdf_bytes, dpi=200)

        results = {}

        if len(pages) >= 1:
            img1 = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            results.update(extract_rows_geometry_center_locked(img1, PAGE_1_KEYS))

        if len(pages) >= 2:
            img2 = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)
            results.update(extract_rows_geometry_center_locked(img2, PAGE_2_KEYS))

        # Deterministic safeguard fill
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
