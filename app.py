import os
import io
import base64
import traceback
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "dev-key")

# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return "HSV Preprocess Service Running v14", 200


# ═══════════════════════════════════════════════════════════════
# DISEASE ENGINE v14 — POSITIONAL + SOLID FILL ISOLATION
# ═══════════════════════════════════════════════════════════════

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

# ──────────────────────────────────────────────────────────────
# DETERMINISTIC PAGE SELECTION (LAST 2 PAGES)
# ──────────────────────────────────────────────────────────────

def find_disease_pages(doc):
    total_pages = len(doc)
    if total_pages >= 2:
        return [total_pages - 2, total_pages - 1]
    return []

# ──────────────────────────────────────────────────────────────
# SOLID FILL ISOLATION (IGNORE GRADIENT TAIL)
# ──────────────────────────────────────────────────────────────

def isolate_solid_fill(page_img, x, y, w, h):
    img_h, img_w = page_img.shape[:2]

    x2 = min(x + w, img_w)
    y2 = min(y + h, img_h)

    if x >= img_w or y >= img_h:
        return None

    bar = page_img[y:y2, x:x2]

    hsv = cv2.cvtColor(bar, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(float) / 255.0
    col_sat = sat.mean(axis=0)

    SAT_SOLID_THRESHOLD = 0.25
    MIN_SOLID_WIDTH = 25

    fill_end = 0

    for i, s in enumerate(col_sat):
        if s > SAT_SOLID_THRESHOLD:
            fill_end = i
        else:
            if fill_end > MIN_SOLID_WIDTH:
                break

    if fill_end < MIN_SOLID_WIDTH:
        return None

    solid = bar[:, :fill_end]

    vert_margin = int(solid.shape[0] * 0.2)
    solid = solid[vert_margin:-vert_margin]

    if solid.size == 0:
        return None

    return solid.reshape(-1, 3)

# ──────────────────────────────────────────────────────────────
# CLASSIFICATION
# ──────────────────────────────────────────────────────────────

def classify_fill(pixels):
    if pixels is None or len(pixels) == 0:
        return "none", 20

    hsv = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    h_vals = hsv[:, 0].astype(float) * 2.0
    s_vals = hsv[:, 1].astype(float) / 255.0

    avg_h = float(np.mean(h_vals))
    avg_s = float(np.mean(s_vals))

    # Grey / None
    if avg_s < 0.15:
        return "none", 20

    # Calibrated hue bands
    if avg_h < 10:
        return "severe", 85
    elif 10 <= avg_h < 25:
        return "moderate", 65
    elif 25 <= avg_h < 45:
        return "mild", 50
    else:
        return "none", 20

# ──────────────────────────────────────────────────────────────
# DETECT ENDPOINT
# ──────────────────────────────────────────────────────────────

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
            return jsonify({
                "results": {},
                "engine": "hsv_v14_final",
                "pages_found": 0
            })

        results = {}
        page_images = []

        for idx in disease_pages:
            pix = doc[idx].get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append(np.array(img))

        for i, field in enumerate(DISEASE_FIELDS_ORDERED):
            page_num = 0 if i < 12 else 1
            row = i if i < 12 else i - 12

            if page_num >= len(page_images):
                results[field] = {
                    "risk_label": "none",
                    "progression_percent": 20,
                    "source": "hsv_v14_final"
                }
                continue

            bar_y = PAGE_Y_STARTS[row]

            pixels = isolate_solid_fill(
                page_images[page_num],
                BAR_X,
                bar_y,
                BAR_W,
                BAR_H
            )

            label, pct = classify_fill(pixels)

            results[field] = {
                "risk_label": label,
                "progression_percent": pct,
                "source": "hsv_v14_final"
            }

        doc.close()

        return jsonify({
            "results": results,
            "engine": "hsv_v14_final",
            "pages_found": len(disease_pages)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ──────────────────────────────────────────────────────────────
# CALIBRATION ENDPOINT (FOR VISUAL DEBUG)
# ──────────────────────────────────────────────────────────────

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

    for page_num, idx in enumerate(disease_pages):
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
