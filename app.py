import os
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")
ENGINE_NAME = "hsv_v18_deterministic"

# =====================================================
# LOCKED VERTICAL ANCHORS (Based on your screenshots)
# Values are ratios of full page height
# =====================================================

PAGE_1_ANCHORS = {
    "large_artery_stiffness": 0.32,
    "peripheral_vessels": 0.36,
    "blood_glucose_uncontrolled": 0.40,
    "small_medium_artery_stiffness": 0.44,
    "atherosclerosis": 0.48,
    "ldl_cholesterol": 0.52,
    "lv_hypertrophy": 0.56,
    "metabolic_syndrome": 0.62,
    "insulin_resistance": 0.66,
    "beta_cell_function_decreased": 0.70,
    "tissue_inflammatory_process": 0.74,
}

PAGE_2_ANCHORS = {
    "hypothyroidism": 0.28,
    "hyperthyroidism": 0.32,
    "hepatic_fibrosis": 0.36,
    "chronic_hepatitis": 0.40,
    "respiratory": 0.48,
    "kidney_function": 0.52,
    "digestive_disorders": 0.56,
    "major_depression": 0.64,
    "adhd_children_learning": 0.68,
    "cerebral_dopamine_decreased": 0.72,
    "cerebral_serotonin_decreased": 0.76,
}

ALL_KEYS = set(PAGE_1_ANCHORS.keys()) | set(PAGE_2_ANCHORS.keys())

@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running v18", 200


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


def process_page(image, anchor_map):
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

    results = {}

    for seg in segments:
        mid_local = (seg[0] + seg[1]) // 2
        mid_absolute = y1 + mid_local
        mid_ratio = mid_absolute / h

        # find closest anchor
        closest_key = None
        closest_dist = 999

        for key, anchor_ratio in anchor_map.items():
            dist = abs(mid_ratio - anchor_ratio)
            if dist < closest_dist:
                closest_dist = dist
                closest_key = key

        if closest_key:
            crop = hsv[seg[0]:seg[1], :, :]
            label, percent = classify_bar(crop)

            results[closest_key] = {
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
            results.update(process_page(img1, PAGE_1_ANCHORS))

        if len(pages) >= 2:
            img2 = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2BGR)
            results.update(process_page(img2, PAGE_2_ANCHORS))

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
