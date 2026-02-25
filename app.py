from flask import Flask, request, jsonify
import fitz
import numpy as np
import cv2

app = Flask(__name__)

ENGINE_NAME = "v98_horizontal_density_anchored_engine"
API_KEY = "ithrive_secure_2026_key"

PAGE_INDEX = 1

# Vertical crop (removes Homeostasis)
TOP_CROP_RATIO = 0.32
BOTTOM_CROP_RATIO = 0.05

# Stripe detection thresholds
SAT_MASK_THRESHOLD = 40
VAL_MASK_THRESHOLD = 40
HORIZONTAL_DENSITY_THRESHOLD = 0.02
COLUMN_DENSITY_THRESHOLD = 0.25
MIN_STRIPE_WIDTH_RATIO = 0.08
MAX_STRIPE_WIDTH_RATIO = 0.85

EXPECTED_ROWS = 24

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
    region = hsv[top_crop:bottom_crop, :]

    sat = region[:, :, 1]
    val = region[:, :, 2]

    colored_mask = (sat > SAT_MASK_THRESHOLD) & (val > VAL_MASK_THRESHOLD)

    # ---- Horizontal density per row ----
    horizontal_density = np.sum(colored_mask, axis=1) / region.shape[1]

    candidate_rows = np.where(horizontal_density > HORIZONTAL_DENSITY_THRESHOLD)[0]

    if len(candidate_rows) == 0:
        return jsonify({"engine": ENGINE_NAME, "error": "No stripe rows detected"})

    # Group contiguous Y indices
    row_groups = np.split(candidate_rows,
                          np.where(np.diff(candidate_rows) != 1)[0] + 1)

    # Filter small noise bands
    row_groups = [g for g in row_groups if len(g) > 5]

    # Sort top → bottom
    row_groups = sorted(row_groups, key=lambda g: g[0])

    # Take strongest 24 bands
    row_groups = row_groups[:EXPECTED_ROWS]

    results = {}

    for i in range(EXPECTED_ROWS):

        if i >= len(row_groups):
            results[DISEASES[i]] = {"risk_label": "none", "source": ENGINE_NAME}
            continue

        y_indices = row_groups[i]
        row_band = region[y_indices[0]:y_indices[-1], :]

        sat_band = row_band[:, :, 1]
        val_band = row_band[:, :, 2]

        colored_band = (sat_band > SAT_MASK_THRESHOLD) & (val_band > VAL_MASK_THRESHOLD)

        col_density = np.sum(colored_band, axis=0) / row_band.shape[0]
        stripe_cols = np.where(col_density > COLUMN_DENSITY_THRESHOLD)[0]

        if len(stripe_cols) == 0:
            results[DISEASES[i]] = {"risk_label": "none", "source": ENGINE_NAME}
            continue

        groups = np.split(stripe_cols,
                          np.where(np.diff(stripe_cols) != 1)[0] + 1)

        largest_group = max(groups, key=len)

        stripe_width_ratio = len(largest_group) / w_img

        if not (MIN_STRIPE_WIDTH_RATIO <= stripe_width_ratio <= MAX_STRIPE_WIDTH_RATIO):
            results[DISEASES[i]] = {"risk_label": "none", "source": ENGINE_NAME}
            continue

        stripe_pixels = row_band[:, largest_group][colored_band[:, largest_group]]

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
        "rows_detected": len(row_groups),
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
