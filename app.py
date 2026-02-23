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

# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running", 200


# ══════════════════════════════════════════════════════════════════
# DISEASE BAR ENGINE v7.1 (HSV DETERMINISTIC)
# ══════════════════════════════════════════════════════════════════

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

CALIBRATION_FIELDS = {
    "large_artery_stiffness",
    "small_medium_artery_stiffness",
    "hepatic_fibrosis"
}


def find_disease_pages(doc):
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if re.search(r"Diseases?\s+and\s+disorder\s+screening", text, re.IGNORECASE):
            pages.append(i)
    return pages


def sample_bar_pixels(page_img, x, y, w, h, sample_pct=0.20):
    img_h, img_w = page_img.shape[:2]

    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)

    if x >= img_w or y >= img_h:
        return np.array([])

    cropped = page_img[y:y2, x:x2]

    sample_w = max(1, int(cropped.shape[1] * sample_pct))
    fill_region = cropped[:, :sample_w]

    vert_margin = int(fill_region.shape[0] * 0.2)
    center_band = fill_region[vert_margin:-vert_margin] if vert_margin > 0 else fill_region

    pixels = center_band.reshape(-1, 3)

    if len(pixels) > 50:
        indices = np.linspace(0, len(pixels) - 1, 50, dtype=int)
        pixels = pixels[indices]

    return pixels


def classify_bar_hsv(pixels_rgb):
    if pixels_rgb.size == 0:
        return {"risk_label": "none", "progression_percent": 20,
                "avg_h": 0, "avg_s": 0, "avg_v": 0}

    hsv = cv2.cvtColor(pixels_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    h_vals = hsv[:, 0].astype(float) * 2.0
    s_vals = hsv[:, 1].astype(float) / 255.0
    v_vals = hsv[:, 2].astype(float) / 255.0

    avg_h = float(np.mean(h_vals))
    avg_s = float(np.mean(s_vals))
    avg_v = float(np.mean(v_vals))

    # UPDATED THRESHOLD
    if avg_s < 0.28:
        return {"risk_label": "none", "progression_percent": 20,
                "avg_h": avg_h, "avg_s": avg_s, "avg_v": avg_v}

    if avg_h <= 24 or avg_h >= 336:
        label, pct = "severe", 85
    elif 25 <= avg_h <= 44:
        label, pct = "moderate", 65
    elif 45 <= avg_h <= 65:
        label, pct = "mild", 50
    else:
        label, pct = "none", 20

    return {"risk_label": label, "progression_percent": pct,
            "avg_h": avg_h, "avg_s": avg_s, "avg_v": avg_v}


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
            return jsonify({"results": {}, "engine": "hsv_v7.1", "pages_found": 0})

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
                                  "source": "hsv_v7.1"}
                continue

            bar_y = PAGE_Y_STARTS[row]
            pixels = sample_bar_pixels(page_images[page_num],
                                       BAR_X, bar_y, BAR_W, BAR_H)

            classification = classify_bar_hsv(pixels)

            if field in CALIBRATION_FIELDS:
                print("HSV CALIBRATION DEBUG")
                print("field:", field)
                print("avg_h:", round(classification["avg_h"], 1))
                print("avg_s:", round(classification["avg_s"], 3))
                print("avg_v:", round(classification["avg_v"], 3))
                print("pixel_count:", len(pixels))
                print("----")

            results[field] = {
                "risk_label": classification["risk_label"],
                "progression_percent": classification["progression_percent"],
                "source": "hsv_v7.1"
            }

        doc.close()

        return jsonify({
            "results": results,
            "engine": "hsv_v7.1",
            "pages_found": len(disease_pages)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# CALIBRATION ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.route("/v1/calibrate-disease-bars", methods=["POST"])
def calibrate_disease_bars():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 401

    if auth_header.replace("Bearer ", "") != API_KEY:
        return jsonify({"error": "Invalid API Key"}), 403

    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    import fitz
    pdf_bytes = request.files["file"].read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    disease_pages = find_disease_pages(doc)
    crops = {}

    for page_num, idx in enumerate(disease_pages[:2]):
        pix = doc[idx].get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_img = np.array(img)

        start_idx = 0 if page_num == 0 else 12

        for row, y in enumerate(PAGE_Y_STARTS):
            field_idx = start_idx + row
            if field_idx >= len(DISEASE_FIELDS_ORDERED):
                break

            field = DISEASE_FIELDS_ORDERED[field_idx]
            y2 = min(y + BAR_H, page_img.shape[0])
            x2 = min(BAR_X + BAR_W, page_img.shape[1])

            crop = page_img[y:y2, BAR_X:x2]
            buf = io.BytesIO()
            Image.fromarray(crop).save(buf, format="PNG")

            crops[field] = base64.b64encode(buf.getvalue()).decode()

    doc.close()
    return jsonify({"crops": crops})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
