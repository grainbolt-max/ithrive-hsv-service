from flask import Flask, request, jsonify
import fitz
import numpy as np
import cv2

app = Flask(__name__)

ENGINE_NAME = "v103_image_extracted_auto_band_detection_classifier"
API_KEY = "ithrive_secure_2026_key"
PAGE_INDEX = 1
EXPECTED_ROWS = 12

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
    if (h < 10 or h > 170) and s > 120:
        return "severe"
    if 10 <= h <= 25 and s > 120:
        return "moderate"
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


def detect_horizontal_bands(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # saturation mask
    sat_mask = hsv[:, :, 1] > 80

    # horizontal projection
    row_sums = sat_mask.sum(axis=1)

    bands = []
    in_band = False
    start = 0

    for i, val in enumerate(row_sums):
        if val > 50 and not in_band:
            in_band = True
            start = i
        elif val <= 50 and in_band:
            in_band = False
            end = i
            if end - start > 5:
                bands.append((start, end))

    # sort by height (largest bands)
    bands = sorted(bands, key=lambda x: x[1] - x[0], reverse=True)

    return bands[:EXPECTED_ROWS]


def classify_panel(img):
    h, w, _ = img.shape
    bands = detect_horizontal_bands(img)

    # sort top to bottom
    bands = sorted(bands, key=lambda x: x[0])

    results = []

    for (y1, y2) in bands:
        band = img[y1:y2, int(w * 0.2):int(w * 0.8)]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)

        mask = hsv[:, :, 1] > 80
        if np.count_nonzero(mask) < 50:
            results.append("none")
            continue

        colored_pixels = hsv[mask]
        avg = colored_pixels.mean(axis=0)
        h_val, s_val, v_val = avg

        severity = classify_hsv(h_val, s_val, v_val)
        results.append(severity)

    # pad if fewer detected
    while len(results) < EXPECTED_ROWS:
        results.append("none")

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
