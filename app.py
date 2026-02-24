import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "hsv_v43_geometry_first_locked"
API_KEY = "ithrive_secure_2026_key"


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


def detect_yellow_span(track_roi):
    hsv = cv2.cvtColor(track_roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 90, 90])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return 0

    span = xs.max() - xs.min()
    percent = int((span / track_roi.shape[1]) * 100)

    return min(percent, 100)


def find_horizontal_tracks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Dilate horizontally to merge bar edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tracks = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # LONG horizontal rectangles only
        if w > image.shape[1] * 0.4 and h > 8:
            roi = image[y:y+h, x:x+w]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_sat = np.mean(hsv[:, :, 1])

            # Confirm it's mostly gray background
            if avg_sat < 60:
                tracks.append((x, y, w, h))

    tracks = sorted(tracks, key=lambda t: t[1])
    return tracks


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


@app.route("/")
def home():
    return "HSV Preprocess Service Running v43"


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

    results = {}
    page_count = 0

    for page in pages:
        page_count += 1

        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        tracks = find_horizontal_tracks(image)

        for i, track in enumerate(tracks):
            if i >= len(DISEASE_KEYS):
                break

            x, y, w, h = track
            roi = image[y:y+h, x:x+w]

            percent = detect_yellow_span(roi)

            results[DISEASE_KEYS[i]] = {
                "progression_percent": percent,
                "risk_label": risk_label(percent),
                "source": ENGINE_NAME
            }

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": page_count,
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
