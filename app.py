from flask import Flask, request, jsonify
import fitz
import numpy as np
import cv2

app = Flask(__name__)

ENGINE_NAME = "v105_row_first_stripe_isolation_production_classifier"
API_KEY = "ithrive_secure_2026_key"
PAGE_INDEX = 1
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
    if (h < 10 or h > 170) and s > 120:
        return "severe"
    if 10 <= h <= 25 and s > 120:
        return "moderate"
    if 25 < h <= 40 and s > 100:
        return "mild"
    return "none"


def extract_image_by_index(page, image_index):
    images = page.get_images(full=True)
    xref = images[image_index][0]
    base_image = page.parent.extract_image(xref)
    image_bytes = base_image["image"]
    np_img = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)


def classify_panel(panel_img):
    h, w, _ = panel_img.shape
    row_height = h // ROWS_PER_PANEL
    results = []

    for i in range(ROWS_PER_PANEL):
        y1 = i * row_height
        y2 = (i + 1) * row_height
        row = panel_img[y1:y2, :]

        hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
        sat_mask = hsv[:, :, 1] > 80

        if np.count_nonzero(sat_mask) < 20:
            results.append("none")
            continue

        # Collapse vertically to detect horizontal stripe cluster
        col_sums = sat_mask.sum(axis=0)

        # Find contiguous region with saturation
        in_cluster = False
        start = 0
        clusters = []

        for j, val in enumerate(col_sums):
            if val > 5 and not in_cluster:
                in_cluster = True
                start = j
            elif val <= 5 and in_cluster:
                in_cluster = False
                end = j
                if end - start > 10:
                    clusters.append((start, end))

        if not clusters:
            results.append("none")
            continue

        # Choose widest cluster (actual stripe)
        cluster = max(clusters, key=lambda x: x[1] - x[0])
        x1, x2 = cluster

        stripe = row[:, x1:x2]
        hsv_stripe = cv2.cvtColor(stripe, cv2.COLOR_BGR2HSV)

        mask = hsv_stripe[:, :, 1] > 80
        if np.count_nonzero(mask) < 20:
            results.append("none")
            continue

        colored_pixels = hsv_stripe[mask]
        avg = colored_pixels.mean(axis=0)
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

    page = doc[PAGE_INDEX]

    # image 1 = homeostasis
    # image 2 = disease rows 1-12
    # image 3 = disease rows 13-24
    panel1 = extract_image_by_index(page, 1)
    panel2 = extract_image_by_index(page, 2)

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
        "results": final_results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
