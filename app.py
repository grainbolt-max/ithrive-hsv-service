from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "v92_calibrated_production_classifier"
API_KEY = "ithrive_secure_2026_key"

TARGET_PAGE_INDEX = 1
TOP_CROP_RATIO = 0.32
BOTTOM_CROP_RATIO = 0.06
ROW_COUNT = 24
LEFT_SCAN_RATIO = 0.65

# Stripe detection
SAT_THRESHOLD = 100
VAL_THRESHOLD = 60
COLUMN_DENSITY_THRESHOLD = 0.20

# Width guards
MIN_WIDTH_RATIO = 0.15
MAX_WIDTH_RATIO = 0.80

# Center-band lock
CENTER_TOP_RATIO = 0.30
CENTER_BOTTOM_RATIO = 0.70

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

    page = pages[TARGET_PAGE_INDEX]
    page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    h, w, _ = page_img.shape

    top_crop = int(h * TOP_CROP_RATIO)
    bottom_crop = int(h * (1 - BOTTOM_CROP_RATIO))
    disease_region = page_img[top_crop:bottom_crop, :]

    region_h, region_w, _ = disease_region.shape
    row_height = region_h // ROW_COUNT
    scan_width = int(region_w * LEFT_SCAN_RATIO)

    results = {}

    for i in range(ROW_COUNT):

        y1 = i * row_height

        center_top = int(y1 + row_height * CENTER_TOP_RATIO)
        center_bottom = int(y1 + row_height * CENTER_BOTTOM_RATIO)

        row_img = disease_region[center_top:center_bottom, :scan_width]

        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        colored_mask = (sat > SAT_THRESHOLD) & (val > VAL_THRESHOLD)

        col_density = np.sum(colored_mask, axis=0) / row_img.shape[0]
        stripe_cols = np.where(col_density > COLUMN_DENSITY_THRESHOLD)[0]

        stripe_width_ratio = 0.0
        severity = "none"

        if len(stripe_cols) > 0:
            stripe_width_ratio = (stripe_cols[-1] - stripe_cols[0]) / scan_width

            if MIN_WIDTH_RATIO <= stripe_width_ratio <= MAX_WIDTH_RATIO:

                valid = colored_mask > 0
                if np.sum(valid) > 0:

                    avg_h = float(np.mean(hsv[:, :, 0][valid]))
                    avg_s = float(np.mean(hsv[:, :, 1][valid]))
                    avg_v = float(np.mean(hsv[:, :, 2][valid]))

                    # 🔒 Grey rejection guard
                    if avg_s < 80 or avg_v > 210:
                        severity = "none"

                    else:
                        # 🔒 Red wraparound detection
                        if (0 <= avg_h <= 10) or (170 <= avg_h <= 179):
                            severity = "severe"
                        elif 10 < avg_h <= 25:
                            severity = "moderate"
                        elif 25 < avg_h <= 45:
                            severity = "mild"
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
