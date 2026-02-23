import os
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")
ENGINE_NAME = "hsv_v21_stripe_locked"

# =========================
# FIXED DISEASE ORDER
# =========================

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


# =========================
# HEALTH CHECK
# =========================

@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running v21", 200


# =========================
# STRIPE SEGMENT DETECTION
# =========================

def detect_stripe_segments(image):
    h, w, _ = image.shape

    # Crop the disease score column
    x1 = int(w * 0.48)
    x2 = int(w * 0.85)
    y1 = int(h * 0.18)
    y2 = int(h * 0.88)

    col = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)

    # Detect vertical stripe texture using Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_strength = np.abs(sobelx)

    # Average vertical edge energy per row
    row_energy = np.mean(edge_strength, axis=1)

    # Stripe detection threshold (stable for template)
    mask = row_energy > 12

    segments = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start > 8:
                segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(mask)))

    # Sort top → bottom
    segments = sorted(segments, key=lambda x: x[0])

    return col, segments


# =========================
# COLOR CLASSIFICATION
# =========================

def classify_color(bgr_crop):
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]

    # Ignore very low saturation pixels (pure gray background)
    mask = sat > 20
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


# =========================
# PAGE PROCESSING
# =========================

def process_page(image, key_order):
    col, segments = detect_stripe_segments(image)

    results = {}

    for idx, seg in enumerate(segments):
        if idx >= len(key_order):
            break

        key = key_order[idx]

        crop = col[seg[0]:seg[1], :]
        label, percent = classify_color(crop)

        results[key] = {
            "progression_percent": percent,
            "risk_label": label,
            "source": ENGINE_NAME
        }

    return results


# =========================
# MAIN API ENDPOINT
# =========================

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
            results.update(process_page(img1, PAGE_1_KEYS))

        if len(pages) >= 2:
            img2 = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)
            results.update(process_page(img2, PAGE_2_KEYS))

        # Fill any missing diseases as none
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
