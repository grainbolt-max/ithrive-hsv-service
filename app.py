import os
import io
import json
import base64
import traceback
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")

ENGINE_NAME = "hsv_v15_dynamic"

# ===============================
# HEALTH CHECK
# ===============================
@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running v15", 200


# ===============================
# CONFIG
# ===============================
BAR_X_RATIO = 0.28        # % of page width where bars begin
BAR_W_RATIO = 0.62        # % of page width bar spans
BAR_HEIGHT = 32
BAR_SPACING = 70

NONE_THRESHOLD = 0.12
MILD_THRESHOLD = 0.25
MOD_THRESHOLD = 0.40


# ===============================
# UTILITIES
# ===============================
def decode_pdf(file_storage):
    pdf_bytes = file_storage.read()
    images = []
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(np.array(img))
    return images


def find_first_bar_row(page_img):
    hsv = cv2.cvtColor(page_img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1] / 255.0

    mask = sat > 0.35
    row_strength = mask.sum(axis=1)

    for y in range(len(row_strength)):
        if row_strength[y] > 300:
            return y

    return None


def classify_bar(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1] / 255.0
    avg_sat = sat.mean()

    if avg_sat < NONE_THRESHOLD:
        return "none", 20
    elif avg_sat < MILD_THRESHOLD:
        return "mild", 50
    elif avg_sat < MOD_THRESHOLD:
        return "moderate", 65
    else:
        return "severe", 85


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

        pages = decode_pdf(request.files["file"])

        results = {}

        for page in pages:
            h, w = page.shape[:2]

            first_bar_y = find_first_bar_row(page)
            if first_bar_y is None:
                continue

            BAR_X = int(w * BAR_X_RATIO)
            BAR_W = int(w * BAR_W_RATIO)

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

            for i, key in enumerate(disease_keys):
                y = first_bar_y + (i * BAR_SPACING)

                if y + BAR_HEIGHT >= h:
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
