import os
import io
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
    "large_artery_stiffness","peripheral_vessels","bp_uncontrolled",
    "small_medium_artery_stiffness","atherosclerosis","ldl_cholesterol",
    "lv_hypertrophy","metabolic_syndrome","insulin_resistance",
    "beta_cell_function_decreased","blood_glucose_uncontrolled",
    "tissue_inflammatory_process","hypothyroidism","hyperthyroidism",
    "hepatic_fibrosis","chronic_hepatitis","prostate_cancer",
    "respiratory","kidney_function","digestive_disorders",
    "major_depression","adhd_children_learning",
    "cerebral_dopamine_decreased","cerebral_serotonin_decreased",
]

BAR_X = 1100
BAR_W = 1200
BAR_H = 50

PAGE_Y_STARTS = [
    600,688,776,864,952,1040,
    1128,1216,1304,1392,1480,1568
]


def find_disease_pages(doc):
    total = len(doc)
    if total >= 2:
        return [total - 2, total - 1]
    elif total == 1:
        return [0]
    return []


def isolate_fill_and_classify(page_img, x, y, w, h):

    img_h, img_w = page_img.shape[:2]
    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)

    bar = page_img[y:y2, x:x2]
    if bar.size == 0:
        return {"risk_label": "none", "progression_percent": 20}

    # Convert to grayscale luminance
    gray = cv2.cvtColor(bar, cv2.COLOR_RGB2GRAY)

    # Background reference = rightmost 15%
    bg_start = int(gray.shape[1] * 0.85)
    bg_region = gray[:, bg_start:]
    bg_mean = np.mean(bg_region)

    # Column brightness
    col_means = np.mean(gray, axis=0)

    # Detect fill where brightness differs from background
    DIFF_THRESHOLD = 8  # tuned for subtle pastel bars
    fill_cols = np.where(np.abs(col_means - bg_mean) > DIFF_THRESHOLD)[0]

    if len(fill_cols) == 0:
        return {"risk_label": "none", "progression_percent": 20}

    fill_start = fill_cols[0]
    fill_end = fill_cols[-1]

    vert_margin = int(bar.shape[0] * 0.2)
    fill_region = bar[vert_margin:-vert_margin, fill_start:fill_end]

    if fill_region.size == 0:
        return {"risk_label": "none", "progression_percent": 20}

    hsv = cv2.cvtColor(fill_region, cv2.COLOR_RGB2HSV)
    avg_h = float(np.mean(hsv[:,:,0])) * 2.0

    print("DEBUG hue:", round(avg_h,1))

    if avg_h <= 30 or avg_h >= 330:
        return {"risk_label": "severe", "progression_percent": 85}
    elif 31 <= avg_h <= 50:
        return {"risk_label": "moderate", "progression_percent": 65}
    elif 51 <= avg_h <= 85:
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

        results = {}
        page_images = []

        for idx in disease_pages:
            pix = doc[idx].get_pixmap(dpi=300)
            img = Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
            page_images.append(np.array(img))

        for i, field in enumerate(DISEASE_FIELDS_ORDERED):

            page_num = 0 if i < 12 else 1
            row = i if i < 12 else i - 12

            bar_y = PAGE_Y_STARTS[row]

            classification = isolate_fill_and_classify(
                page_images[page_num],
                BAR_X, bar_y, BAR_W, BAR_H
            )

            results[field] = {
                "risk_label": classification["risk_label"],
                "progression_percent": classification["progression_percent"],
                "source": "hsv_v11_luminance"
            }

        doc.close()

        return jsonify({
            "results": results,
            "engine": "hsv_v11_luminance",
            "pages_found": len(disease_pages)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
