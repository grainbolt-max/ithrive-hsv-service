from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "v88_final_locked_production_classifier"
API_KEY = "ithrive_secure_2026_key"

TARGET_PAGE_INDEX = 1
TOP_CROP_RATIO = 0.32
BOTTOM_CROP_RATIO = 0.06
ROW_COUNT = 24
LEFT_SCAN_RATIO = 0.65

SAT_THRESHOLD = 150
VAL_THRESHOLD = 60
COLUMN_DENSITY_THRESHOLD = 0.20
MAX_GAP_COLUMNS = 2

MIN_VALID_WIDTH = 0.15
ARTIFACT_WIDTH_REJECT = 0.80

DISEASE_KEYS = [
    "large_artery_stiffness",
    "peripheral_vessel",
    "blood_pressure_uncontrolled",
    "small_medium_artery_stiffness",
    "atherosclerosis",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "diabetes",
    "metabolic_syndrome",
    "insulin_resistance",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "tissue_inflammatory_process",
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
    "cerebral_dopamine_serotonin"
]


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_bytes = request.files["file"].read()
    pages = convert_from_bytes(pdf_bytes, dpi=200)

    if TARGET_PAGE_INDEX >= len(pages):
        return jsonify({"error": "Target page not found"}), 400

    page = pages[TARGET_PAGE_INDEX]
    page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    h, w, _ = page_img.shape

    top_crop = int(h * TOP_CROP_RATIO)
    bottom_crop = int(h * (1 - BOTTOM_CROP_RATIO))
    disease_region = page_img[top_crop:bottom_crop, :]

    region_h, region_w, _ = disease_region.shape
    row_height = region_h // ROW_COUNT

    results = {}

    for i in range(ROW_COUNT):

        y1 = i * row_height
        y2 = (i + 1) * row_height if i < ROW_COUNT - 1 else region_h
        row_img = disease_region[y1:y2, :]

        scan_width = int(region_w * LEFT_SCAN_RATIO)
        scan_region = row_img[:, :scan_width]

        hsv = cv2.cvtColor(scan_region, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        colored_mask = (sat > SAT_THRESHOLD) & (val > VAL_THRESHOLD)

        stripe_end_col = 0
        gap_counter = 0
        stripe_active = False

        for col in range(scan_width):
            col_density = np.sum(colored_mask[:, col]) / len(colored_mask[:, col])
            if col_density > COLUMN_DENSITY_THRESHOLD:
                stripe_active = True
                stripe_end_col = col
                gap_counter = 0
            else:
                if stripe_active:
                    gap_counter += 1
                    if gap_counter > MAX_GAP_COLUMNS:
                        break

        stripe_width_ratio = stripe_end_col / scan_width if stripe_active else 0.0

        severity = "none"

        # 🔒 LOCKED WIDTH GUARDS
        if stripe_width_ratio == 0.0:
            severity = "none"

        elif stripe_width_ratio < MIN_VALID_WIDTH:
            severity = "none"

        elif stripe_width_ratio > ARTIFACT_WIDTH_REJECT:
            severity = "none"

        else:
            stripe_pixels = colored_mask[:, :stripe_end_col]
            stripe_hsv = hsv[:, :stripe_end_col]
            valid = stripe_pixels > 0

            if np.sum(valid) > 0:
                avg_h = float(np.mean(stripe_hsv[:, :, 0][valid]))

                if 0 <= avg_h <= 10:
                    severity = "severe"
                elif 10 < avg_h <= 22:
                    severity = "moderate"
                elif 22 < avg_h <= 40:
                    severity = "mild"
                else:
                    severity = "none"
            else:
                severity = "none"

        results[DISEASE_KEYS[i]] = {
            "risk_label": severity,
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "page_index_processed": TARGET_PAGE_INDEX,
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
