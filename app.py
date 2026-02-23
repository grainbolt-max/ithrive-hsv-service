import os
import re
import io
import base64
import traceback
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")

@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running", 200


DISEASE_FIELDS_ORDERED = [
    "large_artery_stiffness",
    "peripheral_vessels",
    "bp_uncontrolled",
    "small_medium_artery_stiffness",
    "atherosclerosis",
    "ldl_cholesterol",
    "lv_hypertrophy",
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
    "respiratory",
    "kidney_function",
    "digestive_disorders",
    "major_depression",
    "adhd_children_learning",
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased",
]

BAR_X = 1100
BAR_W = 1200
BAR_H = 50

PAGE_Y_STARTS = [
    600, 688, 776, 864, 952, 1040,
    1128, 1216, 1304, 1392, 1480, 1568
]


def find_disease_pages(doc):
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if re.search(r"Diseases?\s+and\s+disorder\s+screening", text, re.IGNORECASE):
            pages.append(i)
    return pages


def isolate_fill_and_classify(page_img, x, y, w, h):

    img_h, img_w = page_img.shape[:2]
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)

    if x >= img_w or y >= img_h:
        return {"risk_label": "none", "progression_percent": 20}

    bar = page_img[y:y2, x:x2]
    if bar.size == 0:
        return {"risk_label": "none", "progression_percent": 20}

    hsv_bar = cv2.cvtColor(bar, cv2.COLOR_RGB2HSV)

    h_map = hsv_bar[:, :, 0].astype(float) * 2.0
    s_map = hsv_bar[:, :, 1].astype(float) / 255.0

    # Detect fill columns by saturation
    SAT_FILL_THRESHOLD = 0.10
    col_sat = s_map.mean(axis=0)
    fill_cols = np.where(col_sat > SAT_FILL_THRESHOLD)[0]

    if len(fill_cols) == 0:
        return {"risk_label": "none", "progression_percent": 20}

    fill_start = fill_cols[0]
    fill_end = fill_cols[-1]

    vert_margin = int(bar.shape[0] * 0.2)
    vert_end = bar.shape[0] - vert_margin

    fill_h = h_map[vert_margin:vert_end, fill_start:fill_end]
    fill_s = s_map[vert_margin:vert_end, fill_start:fill_end]

    if fill_h.size == 0:
        return {"risk_label": "none", "progression_percent": 20}

    avg_h = float(np.mean(fill_h))
    avg_s = float(np.mean(fill_s))

    # Grey detection
    if avg_s < 0.07:
        return {"risk_label": "none", "progression_percent": 20}

    # Hue classification
    if avg_h <= 25 or avg_h >= 335:
        return {"risk_label": "severe", "progression_percent": 85}
    elif 26 <= avg_h <= 44:
        return {"risk_label": "moderate", "progression_percent": 65}
    elif 45 <= avg_h <= 70:
        return {"risk_label": "mild", "progression_percent": 50}
    else:
        return {"risk_label": "none", "progression_percent": 20}


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 401

    if auth_header.replace("Bearer ", "") != API_KEY:
        return jsonify({"error": "Invalid API Key"}), 403

    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    pdf_bytes = request.files["file"].read()

    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        disease_pages = find_disease_pages(doc)

        if not disease_pages:
            return jsonify({"results": {}, "engine": "hsv_v8_fill", "pages_found": 0})

        results = {}
        page_images = []

        for idx in disease_pages[:2]:
            pix = doc[idx].get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append(np.array(img))

        for i, field in enumerate(DISEASE_FIELDS_ORDERED):

            page_num = 0 if i < 12 else 1
            row = i if i < 12 else i - 12

            if page_num >= len(page_images):
                results[field] = {"risk_label": "none",
                                  "progression_percent": 20,
                                  "source": "hsv_v8_fill"}
                continue

            bar_y = PAGE_Y_STARTS[row]

            classification = isolate_fill_and_classify(
                page_images[page_num],
                BAR_X,
                bar_y,
                BAR_W,
                BAR_H
            )

            results[field] = {
                "risk_label": classification["risk_label"],
                "progression_percent": classification["progression_percent"],
                "source": "hsv_v8_fill"
            }

        doc.close()

        return jsonify({
            "results": results,
            "engine": "hsv_v8_fill",
            "pages_found": len(disease_pages)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/v1/calibrate-disease-bars", methods=["POST"])
def calibrate_disease_bars():
    return jsonify({"message": "Calibration unchanged"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
