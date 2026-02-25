from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "v89_row_aligned_stripe_engine"
API_KEY = "ithrive_secure_2026_key"

TARGET_PAGE_INDEX = 1
TOP_CROP_RATIO = 0.32
BOTTOM_CROP_RATIO = 0.06
LEFT_SCAN_RATIO = 0.65

SAT_THRESHOLD = 150
VAL_THRESHOLD = 60

MIN_BAND_HEIGHT = 8
MAX_VERTICAL_GAP = 4

MIN_WIDTH_RATIO = 0.15
MAX_WIDTH_RATIO = 0.80

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
    scan_width = int(region_w * LEFT_SCAN_RATIO)
    scan_region = disease_region[:, :scan_width]

    hsv = cv2.cvtColor(scan_region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    colored_mask = (sat > SAT_THRESHOLD) & (val > VAL_THRESHOLD)

    # Collapse vertically to detect horizontal colored rows
    row_density = np.sum(colored_mask, axis=1) / scan_width

    bands = []
    band_start = None
    gap = 0

    for y in range(region_h):
        if row_density[y] > 0.02:
            if band_start is None:
                band_start = y
            gap = 0
        else:
            if band_start is not None:
                gap += 1
                if gap > MAX_VERTICAL_GAP:
                    band_end = y - gap
                    if band_end - band_start >= MIN_BAND_HEIGHT:
                        bands.append((band_start, band_end))
                    band_start = None
                    gap = 0

    # Close final band
    if band_start is not None:
        band_end = region_h - 1
        if band_end - band_start >= MIN_BAND_HEIGHT:
            bands.append((band_start, band_end))

    if len(bands) != 24:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": f"Expected 24 stripe bands, detected {len(bands)}"
        })

    results = {}

    for i, (y1, y2) in enumerate(bands):

        band = scan_region[y1:y2, :]

        hsv_band = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        sat_band = hsv_band[:, :, 1]
        val_band = hsv_band[:, :, 2]

        band_mask = (sat_band > SAT_THRESHOLD) & (val_band > VAL_THRESHOLD)

        col_density = np.sum(band_mask, axis=0) / band_mask.shape[0]

        stripe_cols = np.where(col_density > 0.20)[0]

        if len(stripe_cols) == 0:
            stripe_width_ratio = 0.0
        else:
            stripe_width_ratio = (stripe_cols[-1] - stripe_cols[0]) / scan_width

        severity = "none"

        if stripe_width_ratio >= MIN_WIDTH_RATIO and stripe_width_ratio <= MAX_WIDTH_RATIO:
            valid = band_mask > 0
            if np.sum(valid) > 0:
                avg_h = float(np.mean(hsv_band[:, :, 0][valid]))

                if 0 <= avg_h <= 10:
                    severity = "severe"
                elif 10 < avg_h <= 22:
                    severity = "moderate"
                elif 22 < avg_h <= 40:
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
