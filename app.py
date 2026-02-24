import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "hsv_v51_contour_locked_deterministic"
API_KEY = "ithrive_secure_2026_key"

DISEASE_KEYS = [
    "adhd_children_learning",
    "atherosclerosis",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "blood_pressure_uncontrolled",
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased",
    "chronic_hepatitis",
    "digestive_disorders",
    "hepatic_fibrosis",
    "hyperthyroidism",
    "hypothyroidism",
    "insulin_resistance",
    "kidney_function_disorders",
    "large_artery_stiffness",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "major_depression",
    "metabolic_syndrome",
    "peripheral_vessel",
    "prostate_cancer",
    "respiratory_disorders",
    "small_medium_artery_stiffness",
    "tissue_inflammatory_process"
]


def risk_label(percent):
    if percent >= 75:
        return "severe"
    elif percent >= 50:
        return "moderate"
    elif percent >= 20:
        return "mild"
    elif percent > 0:
        return "normal"
    else:
        return "none"


def detect_track_bands(image):
    """
    Detect horizontal track rectangles using contour geometry.
    """

    h, w, _ = image.shape

    # Crop right panel
    panel = image[int(h * 0.35):int(h * 0.92),
                  int(w * 0.45):int(w * 0.98)]

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    # Emphasize horizontal shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Filter by geometry
        if cw > panel.shape[1] * 0.6 and 10 < ch < 35:
            center_y = y + ch // 2
            centers.append(center_y)

    # Convert back to full image coordinates
    offset_y = int(h * 0.35)
    centers = [c + offset_y for c in centers]

    return sorted(centers)


def measure_yellow_span(image, center_y):
    h, w, _ = image.shape

    band_height = 20
    y1 = max(center_y - band_height // 2, 0)
    y2 = min(center_y + band_height // 2, h)

    x1 = int(w * 0.50)
    x2 = int(w * 0.98)

    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 110, 120])
    upper_yellow = np.array([38, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0

    span = xs.max() - xs.min()
    percent = int((span / roi.shape[1]) * 100)

    if percent < 10:
        return 0

    return min(percent, 100)


@app.route("/")
def home():
    return "HSV Preprocess Service Running v51"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception:
        return jsonify({"error": "PDF conversion failed"}), 500

    page = pages[1]
    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    band_centers = detect_track_bands(image)

    results = {}

    for i, center_y in enumerate(band_centers):
        if i >= len(DISEASE_KEYS):
            break

        percent = measure_yellow_span(image, center_y)

        results[DISEASE_KEYS[i]] = {
            "progression_percent": percent,
            "risk_label": risk_label(percent),
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": len(pages),
        "bands_detected": len(band_centers),
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
