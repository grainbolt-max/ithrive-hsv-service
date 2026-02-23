import os
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")
ENGINE_NAME = "hsv_v20_order_locked"

# Fixed disease order (confirmed stable layout)

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
    "tissue_inflammatory_process",
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
    "cerebral_serotonin_decreased",
]

ALL_KEYS = set(PAGE_1_KEYS) | set(PAGE_2_KEYS)


@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running v20", 200


def classify_bar(hsv_crop):
    hue = hsv_crop[:, :, 0]
    sat = hsv_crop[:, :, 1]

    mask = sat > 30
    if np.sum(mask) == 0:
        return "none", 20

    avg_hue = np.mean(hue[mask])

    if 20 <= avg_hue <= 35:
        return "mild", 50
    elif 0 <= avg_hue <= 10:
        return "severe", 85
    elif 45 <= avg_hue <= 80:
        return "normal", 10
    else:
        return "none", 20


def detect_segments(image):
    h, w, _ = image.shape

    x1 = int(w * 0.48)
    x2 = int(w * 0.85)
    y1 = int(h * 0.18)
    y2 = int(h * 0.88)

    col = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    row_strength = np.mean(sat, axis=1)
    mask = row_strength > 25

    segments = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start > 6:
                segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(mask)))

    # sort top to bottom
    segments = sorted(segments, key=lambda x: x[0])

    return hsv, segments


def process_page(image, key_order):
    hsv, segments = detect_segments(image)

    results = {}

    for idx, seg in enumerate(segments):
        if idx >= len(key_order):
            break

        key = key_order[idx]

        crop = hsv[seg[0]:seg[1], :, :]
        label, percent = classify_bar(crop)

        results[key] = {
            "progression_percent": percent,
            "risk_label": label,
            "source": ENGINE_NAME
        }

    return results


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
