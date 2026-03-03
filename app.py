from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import gc

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v2_WITH_MISMATCH_GUARD"
ENGINE_VERSION = "3.3.0_canonical_layout_guard"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(name)

RENDER_DPI = 150
X_LEFT = 704
X_RIGHT = 710

# Deterministic minimum required colored rows to consider layout valid
MIN_REQUIRED_COLOR_ROWS = 5

DISEASE_COORDINATES = {
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


def classify_risk(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    if total_pixels == 0:
        return {"label": "None/Low", "color_ratio": 0.0}

    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = red_mask1 + red_mask2

    orange_mask = cv2.inRange(hsv, (15, 100, 100), (30, 255, 255))
    yellow_mask = cv2.inRange(hsv, (31, 100, 100), (65, 255, 255))

    red_pct = np.count_nonzero(red_mask) / total_pixels
    orange_pct = np.count_nonzero(orange_mask) / total_pixels
    yellow_pct = np.count_nonzero(yellow_mask) / total_pixels

    dominant_ratio = max(red_pct, orange_pct, yellow_pct)

    if dominant_ratio < 0.05:
        return {"label": "None/Low", "color_ratio": dominant_ratio}

    if red_pct >= orange_pct and red_pct >= yellow_pct:
        return {"label": "Severe", "color_ratio": red_pct}

    if orange_pct >= red_pct and orange_pct >= yellow_pct:
        return {"label": "Moderate", "color_ratio": orange_pct}

    if yellow_pct >= red_pct and yellow_pct >= orange_pct:
        return {"label": "Mild", "color_ratio": yellow_pct}

    return {"label": "None/Low", "color_ratio": dominant_ratio}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION,
        "dpi": RENDER_DPI
    })


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    try:
        if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        pdf_bytes = request.files["file"].read()

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

        results = {}
        total_color_hits = 0

        for disease, (y1, y2) in DISEASE_COORDINATES.items():
            roi = page_image[y1:y2, X_LEFT:X_RIGHT]

            if roi.size == 0:
                results[disease] = "None/Low"
                continue

            outcome = classify_risk(roi)
            results[disease] = outcome["label"]

            if outcome["color_ratio"] > 0.05:
                total_color_hits += 1

        # Deterministic layout mismatch gate
        if total_color_hits < MIN_REQUIRED_COLOR_ROWS:
            return jsonify({
                "error": "LAYOUT_MISMATCH",
                "reason": "Insufficient stripe confidence",
                "color_hits": total_color_hits,
                "required": MIN_REQUIRED_COLOR_ROWS
            }), 422

        del page_image
        gc.collect()

        return jsonify({
            "engine": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "results": results
        })

    except Exception as e:
        return jsonify({
            "error": "Internal error",
            "details": str(e)
        }), 500


if name == "main":
    app.run(host="0.0.0.0", port=10000)
