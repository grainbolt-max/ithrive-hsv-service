from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import cv2
import io

app = Flask(__name__)

ENGINE_NAME = "v94_calibrated_hsv_band_production_classifier"
API_KEY = "ithrive_secure_2026_key"

# === Disease Order (24 rows, top → bottom) ===
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

# === Strict center band for disease area only ===
CENTER_X_START_RATIO = 0.32
CENTER_X_END_RATIO   = 0.68

ROW_COUNT = 24


def classify_color(h, s, v):
    """
    Calibrated HSV bands based on actual PDF measurements.
    """

    # Light Grey / None
    if s < 20:
        return "none"

    # Severe (Red)
    if (0 <= h <= 25) and (s > 180) and (50 <= v <= 220):
        return "severe"

    # Moderate (Orange)
    if (20 < h <= 40) and (s > 100) and (60 <= v <= 220):
        return "moderate"

    # Mild (Yellow)
    if (40 < h <= 70) and (s > 120) and (80 <= v <= 240):
        return "mild"

    return "none"


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except:
        return jsonify({"error": "Invalid PDF"}), 400

    # Page 2 contains all 24 disease rows
    page_index = 1
    if len(doc) <= page_index:
        return jsonify({"error": "Page 2 not found"}), 400

    page = doc[page_index]

    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_img, w_img = hsv.shape[:2]

    x_start = int(w_img * CENTER_X_START_RATIO)
    x_end   = int(w_img * CENTER_X_END_RATIO)

    row_height = h_img // ROW_COUNT

    results = {}

    for i in range(ROW_COUNT):

        y_top = i * row_height
        y_bottom = (i + 1) * row_height

        row_slice = hsv[y_top:y_bottom, x_start:x_end]

        if row_slice.size == 0:
            results[DISEASES[i]] = {
                "risk_label": "none",
                "source": ENGINE_NAME
            }
            continue

        # Only count colored pixels (avoid white)
        mask = row_slice[:, :, 1] > 20
        colored_pixels = row_slice[mask]

        if len(colored_pixels) < 50:
            risk = "none"
        else:
            avg_h = np.mean(colored_pixels[:, 0])
            avg_s = np.mean(colored_pixels[:, 1])
            avg_v = np.mean(colored_pixels[:, 2])
            risk = classify_color(avg_h, avg_s, avg_v)

        results[DISEASES[i]] = {
            "risk_label": risk,
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "page_index_processed": page_index,
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
