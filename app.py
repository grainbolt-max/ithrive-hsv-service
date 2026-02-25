from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import cv2

app = Flask(__name__)

ENGINE_NAME = "v101_image_extracted_locked_production_classifier"
API_KEY = "ithrive_secure_2026_key"
PAGE_INDEX = 1  # Page 2 (0-based)
ROWS_PER_PANEL = 12

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


def classify_hsv(h, s, v):
    # HSV thresholds calibrated to your legend

    # light grey (none/low)
    if s < 30 and v > 150:
        return "none"

    # red (severe)
    if (h < 10 or h > 170) and s > 120:
        return "severe"

    # orange (moderate)
    if 10 <= h <= 25 and s > 120:
        return "moderate"

    # yellow (mild)
    if 25 < h <= 40 and s > 100:
        return "mild"

    return "none"


def extract_image_by_index(page, image_index):
    images = page.get_images(full=True)
    if image_index >= len(images):
        return None

    xref = images[image_index][0]
    base_image = page.parent.extract_image(xref)
    image_bytes = base_image["image"]

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img


def classify_panel(img):
    h, w, _ = img.shape
    row_height = h // ROWS_PER_PANEL
    results = []

    for i in range(ROWS_PER_PANEL):
        y1 = i * row_height
        y2 = (i + 1) * row_height

        band = img[y1:y2, int(w * 0.2):int(w * 0.8)]  # center 60%

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        avg = hsv.mean(axis=(0, 1))
        h_val, s_val, v_val = avg

        severity = classify_hsv(h_val, s_val, v_val)
        results.append(severity)

    return results


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
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    if len(doc) <= PAGE_INDEX:
        return jsonify({"error": "Page 2 not found"}), 400

    page = doc[PAGE_INDEX]

    # Image 1 = Homeostasis
    # Image 2 = Disease rows 1–12
    # Image 3 = Disease rows 13–24

    panel1 = extract_image_by_index(page, 1)
    panel2 = extract_image_by_index(page, 2)

    if panel1 is None or panel2 is None:
        return jsonify({"error": "Disease panels not found"}), 400

    results1 = classify_panel(panel1)
    results2 = classify_panel(panel2)

    combined = results1 + results2

    final_results = {}
    for i in range(24):
        final_results[DISEASES[i]] = {
            "risk_label": combined[i],
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "page_index_processed": PAGE_INDEX,
        "results": final_results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
