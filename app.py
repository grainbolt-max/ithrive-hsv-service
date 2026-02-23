import os
import io
import re
import traceback
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")
ENGINE_NAME = "hsv_v16_production"

# ===============================
# HEALTH CHECK
# ===============================
@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running v16", 200


# ===============================
# CONFIG
# ===============================
BAR_X_RATIO = 0.28
BAR_W_RATIO = 0.62
BAR_HEIGHT = 32
BAR_SPACING = 70


# ===============================
# PDF UTILITIES
# ===============================
def load_disease_pages(file_storage):
    import fitz

    pdf_bytes = file_storage.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page in doc:
        text = page.get_text()

        if re.search(r"Disease", text, re.IGNORECASE):
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages.append(np.array(img))

    return pages


# ===============================
# BAR LOCATION
# ===============================
def find_first_bar_row(page_img):
    hsv = cv2.cvtColor(page_img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1] / 255.0

    mask = sat > 0.30
    row_strength = mask.sum(axis=1)

    for y in range(len(row_strength)):
        if row_strength[y] > 400:
            return y

    return None


# ===============================
# COLOR CLASSIFIER
# ===============================
def classify_bar(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0].astype(float) * 2.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    avg_h = float(np.mean(h))
    avg_s = float(np.mean(s))
    avg_v = float(np.mean(v))

    # 1️⃣ None / Grey (very low saturation)
    if avg_s < 0.08:
        return "none", 20

    # 2️⃣ Yellow (mild)
    if 40 <= avg_h <= 70 and avg_s > 0.15:
        return "mild", 50

    # 3️⃣ Orange (moderate)
    if 15 <= avg_h < 40:
        return "moderate", 65

    # 4️⃣ Red (severe)
    if avg_h < 15 or avg_h > 340:
        return "severe", 85

    # fallback
    return "mild", 50


# ===============================
# DETECT ENDPOINT
# ===============================
@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease():
    try:
        auth = request.headers.get("Authorization", "")
        if f"Bearer {API_KEY}" != auth:
            return jsonify({"error": "Unauthorized"}), 401

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        pages = load_disease_pages(request.files["file"])

        results = {}

        disease_keys = [
            "large_artery_stiffness",
            "peripheral_vessels",
            "small_medium_artery_stiffness",
            "atherosclerosis",
            "lv_hypertrophy",
            "ldl_cholesterol",
            "metabolic_syndrome",
            "insulin_resistance",
            "hepatic_fibrosis",
            "chronic_hepatitis",
            "respiratory",
            "bp_uncontrolled",
            "beta_cell_function_decreased",
            "blood_glucose_uncontrolled",
            "tissue_inflammatory_process",
            "prostate_cancer",
            "major_depression",
            "adhd_children_learning",
            "cerebral_dopamine_decreased",
            "cerebral_serotonin_decreased",
            "kidney_function",
            "hyperthyroidism",
            "hypothyroidism",
            "digestive_disorders"
        ]

        for page in pages:
            h_img, w_img = page.shape[:2]

            first_bar_y = find_first_bar_row(page)
            if first_bar_y is None:
                continue

            BAR_X = int(w_img * BAR_X_RATIO)
            BAR_W = int(w_img * BAR_W_RATIO)

            for i, key in enumerate(disease_keys):
                y = first_bar_y + (i * BAR_SPACING)

                if y + BAR_HEIGHT >= h_img:
                    break

                crop = page[y:y+BAR_HEIGHT, BAR_X:BAR_X+BAR_W]

                label, percent = classify_bar(crop)

                results[key] = {
                    "risk_label": label,
                    "progression_percent": percent,
                    "source": ENGINE_NAME
                }

        return jsonify({
            "engine": ENGINE_NAME,
            "pages_found": len(pages),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
