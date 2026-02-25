from flask import Flask, request, jsonify
import fitz
import numpy as np
import cv2

app = Flask(__name__)

ENGINE_NAME = "v97_vertically_anchored_stripe_production_engine"
API_KEY = "ithrive_secure_2026_key"

PAGE_INDEX = 1  # Page 2
ROW_COUNT = 24

# ---- Vertical crop tuned to your PDF layout ----
# Removes homeostasis and top white gap
TOP_CROP_RATIO = 0.36
BOTTOM_CROP_RATIO = 0.06

# ---- Stripe detection thresholds ----
SAT_MASK_THRESHOLD = 40
VAL_MASK_THRESHOLD = 40
COLUMN_DENSITY_THRESHOLD = 0.25
MIN_STRIPE_WIDTH_RATIO = 0.10
MAX_STRIPE_WIDTH_RATIO = 0.80

DISEASES = [
    "adhd_children_learning",
    "atherosclerosis",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "blood_pressure_uncontrolled",
    "cerebral_dopamine_serotonin",
    "chronic_hepatitis",
    "diabetes",
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


def classify_color(h, s, v):

    if s < 20:
        return "none"

    if (0 <= h <= 25 or 170 <= h <= 179) and s > 150:
        return "severe"

    if 20 < h <= 40 and s > 100:
        return "moderate"

    if 40 < h <= 70 and s > 100:
        return "mild"

    return "none"


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():

    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except:
        return jsonify({"error": "Invalid PDF"}), 400

    if len(doc) <= PAGE_INDEX:
        return jsonify({"error": "Page 2 not found"}), 400

    page = doc[PAGE_INDEX]
    pix = page.get_pixmap(dpi=200)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_img, w_img = hsv.shape[:2]

    # ---- Vertical crop ----
    top_crop = int(h_img * TOP_CROP_RATIO)
    bottom_crop = int(h_img * (1 - BOTTOM_CROP_RATIO))
    disease_region = hsv[top_crop:bottom_crop, :]

    region_h = disease_region.shape[0]
    row_height = region_h // ROW_COUNT

    results = {}

    for i in range(ROW_COUNT):

        y1 = i * row_height
        y2 = (i + 1) * row_height

        row = disease_region[y1:y2, :]

        if row.size == 0:
            results[DISEASES[i]] = {"risk_label": "none", "source": ENGINE_NAME}
            continue

        sat = row[:, :, 1]
        val = row[:, :, 2]

        colored_mask = (sat > SAT_MASK_THRESHOLD) & (val > VAL_MASK_THRESHOLD)

        col_density = np.sum(colored_mask, axis=0) / row.shape[0]
        stripe_cols = np.where(col_density > COLUMN_DENSITY_THRESHOLD)[0]

        if len(stripe_cols) == 0:
            results[DISEASES[i]] = {"risk_label": "none", "source": ENGINE_NAME}
            continue

        groups = np.split(stripe_cols, np.where(np.diff(stripe_cols) != 1)[0] + 1)
        largest_group = max(groups, key=len)

        stripe_width_ratio = len(largest_group) / w_img

        if not (MIN_STRIPE_WIDTH_RATIO <= stripe_width_ratio <= MAX_STRIPE_WIDTH_RATIO):
            results[DISEASES[i]] = {"risk_label": "none", "source": ENGINE_NAME}
            continue

        stripe_region = row[:, largest_group]
        stripe_pixels = stripe_region[colored_mask[:, largest_group]]

        if len(stripe_pixels) < 50:
            results[DISEASES[i]] = {"risk_label": "none", "source": ENGINE_NAME}
            continue

        avg_h = float(np.mean(stripe_pixels[:, 0]))
        avg_s = float(np.mean(stripe_pixels[:, 1]))
        avg_v = float(np.mean(stripe_pixels[:, 2]))

        severity = classify_color(avg_h, avg_s, avg_v)

        results[DISEASES[i]] = {
            "risk_label": severity,
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "page_index_processed": PAGE_INDEX,
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
