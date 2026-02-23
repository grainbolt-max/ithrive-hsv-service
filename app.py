from flask import Flask, request, jsonify
import fitz
import cv2
import numpy as np
import os

app = Flask(__name__)

ENGINE_VERSION = "hsv_v29_geometry_left_span_locked"
API_KEY = "ithrive_secure_2026_key"

ROWS_PER_PAGE = 14
PAGE1_HEADING_ROWS = {0, 8}
PAGE2_HEADING_ROWS = {0, 8}

PAGE1_DISEASE_KEYS = [
    "large_artery_stiffness",
    "peripheral_vessel",
    "blood_pressure_uncontrolled",
    "small_medium_artery_stiffness",
    "atherosclerosis",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "metabolic_syndrome",
    "insulin_resistance",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "tissue_inflammatory_process",
]

PAGE2_DISEASE_KEYS = [
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
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased",
]

def map_percent_to_label(percent):
    if percent <= 10:
        return "normal"
    elif percent <= 20:
        return "none"
    elif percent <= 50:
        return "mild"
    elif percent <= 75:
        return "moderate"
    else:
        return "severe"

def render_page(page):
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def detect_overlay_span(row_img):

    height, width, _ = row_img.shape

    # focus on central band vertically
    y1 = int(height * 0.30)
    y2 = int(height * 0.75)
    scan = row_img[y1:y2, :]

    hsv = cv2.cvtColor(scan, cv2.COLOR_BGR2HSV)

    # Color masks
    yellow = cv2.inRange(hsv, (20, 80, 80), (35, 255, 255))
    orange = cv2.inRange(hsv, (10, 100, 100), (20, 255, 255))
    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    neutral = cv2.inRange(hsv, (0, 0, 160), (180, 40, 255))

    mask = yellow | orange | red1 | red2 | neutral

    column_sum = np.sum(mask, axis=0)
    threshold = np.max(column_sum) * 0.3

    overlay_cols = np.where(column_sum > threshold)[0]

    if len(overlay_cols) == 0:
        return 20

    left = overlay_cols[0]
    right = overlay_cols[-1]

    span_ratio = (right - left) / width
    percent = int(round(span_ratio * 100 / 10) * 10)

    percent = max(10, min(percent, 100))

    return percent

@app.route("/")
def home():
    return f"HSV Preprocess Service Running v29"

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_bytes = request.files["file"].read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    results = {}
    target_pages = min(2, len(doc))

    for page_index in range(target_pages):

        page = doc.load_page(page_index)
        img = render_page(page)

        height = img.shape[0]
        row_height = height // ROWS_PER_PAGE

        disease_counter = 0

        for row_index in range(ROWS_PER_PAGE):

            if page_index == 0 and row_index in PAGE1_HEADING_ROWS:
                continue
            if page_index == 1 and row_index in PAGE2_HEADING_ROWS:
                continue

            y1 = row_index * row_height
            y2 = (row_index + 1) * row_height
            row_img = img[y1:y2, :]

            if page_index == 0:
                disease_key = PAGE1_DISEASE_KEYS[disease_counter]
            else:
                disease_key = PAGE2_DISEASE_KEYS[disease_counter]

            percent = detect_overlay_span(row_img)

            results[disease_key] = {
                "progression_percent": percent,
                "risk_label": map_percent_to_label(percent),
                "source": ENGINE_VERSION
            }

            disease_counter += 1

    return jsonify({
        "engine": ENGINE_VERSION,
        "pages_found": target_pages,
        "results": results
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
