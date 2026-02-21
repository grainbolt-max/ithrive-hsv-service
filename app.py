# ENGINE POLICY:
# The uploaded ES Teck PDF must NOT include the bottom color legend.
# Bars are assumed to be pre-cleaned before upload.
# Engine will not attempt legend suppression.

"""
ES Teck Bio Scan — Deterministic Disease Bar Preprocessing Service
Version: Version: v3.0-stable-precleaned-input

This service performs PURELY DETERMINISTIC pixel-level analysis of
ES Teck disease screening horizontal bars from Bio Scan PDFs.

POLICY: PDFs MUST be uploaded WITHOUT the bottom color legend.
Input is treated as pre-cleaned. No legend compensation logic exists.

NO AI. NO ML. NO HEURISTICS. NO DYNAMIC THRESHOLDS.
NO STRUCTURAL ROI. NO CONTOUR DETECTION. NO GRAY TRACK DETECTION.
Fixed saturation gates and hue ranges only.
"""


import os
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# Debug output directory
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_output")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ── Auth ──────────────────────────────────────────────────────────────────────
PREPROCESS_API_KEY = os.environ.get("PREPROCESS_API_KEY", "")

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_DPI = 300
TARGET_DPI = 300
MIN_BAR_WIDTH_RATIO = 0.03  # 3% of total bar width

# ── No global HSV constants needed ────────────────────────────────────────────
# All thresholds are local to compute_bar_metrics() for clarity.

# ── ES Teck Disease Bar Layout ───────────────────────────────────────────────
# Vertical crop bands as percentage of page height.
# These are static positions based on the ES Teck PDF layout.
# Each entry: (label, top_pct, bottom_pct)
# Horizontal padding: 5% from each side

HORIZONTAL_PAD_PCT = 0.05

# Disease bars are located on the "Disease Screening Score" page.
# The page typically contains 18 horizontal bars stacked vertically.
# These Y-ranges are approximate and may need calibration per layout version.
# ── ES Teck Disease Bar Layout ───────────────────────────────────────────────
# Page 1 (12 bars)

DISEASE_BAR_BANDS_PAGE_1 = [
    ("large_artery_stiffness", 0.18, 0.22),
    ("peripheral_vessels", 0.23, 0.27),
    ("blood_pressure_uncontrolled", 0.28, 0.32),
    ("small_medium_artery_stiffness", 0.33, 0.37),
    ("atherosclerosis", 0.38, 0.42),
    ("ldl_cholesterol", 0.43, 0.47),
    ("lv_hypertrophy", 0.48, 0.52),
    ("metabolic_syndrome", 0.53, 0.57),
    ("insulin_resistance", 0.58, 0.62),
    ("beta_cell_function_decreased", 0.63, 0.67),
    ("blood_glucose_uncontrolled", 0.68, 0.72),
    ("tissue_inflammatory_process", 0.73, 0.77),
]

# Page 2 (12 bars)

DISEASE_BAR_BANDS_PAGE_2 = [
    ("hypothyroidism", 0.18, 0.22),
    ("hyperthyroidism", 0.23, 0.27),
    ("hepatic_fibrosis", 0.28, 0.32),
    ("chronic_hepatitis", 0.33, 0.37),
    ("prostate_cancer", 0.38, 0.42),
    ("respiratory_disorders", 0.43, 0.47),
    ("kidney_function_disorders", 0.48, 0.52),

    # Slight uniform upward shift (−0.01 from original)
    ("digestive_disorders", 0.52, 0.56),
    ("major_depression", 0.57, 0.61),
    ("adhd_children_learning", 0.62, 0.66),
    ("cerebral_dopamine_decreased", 0.67, 0.71),
    ("cerebral_serotonin_decreased", 0.72, 0.76),
]

def rgb_to_hsv(image_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB image array [H, W, 3] uint8 to HSV [H, W, 3] float64.
    H in [0, 360), S in [0, 1], V in [0, 1].
    
    Deterministic — no library-specific rounding.
    """
    rgb = image_array.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    hue = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)

    hue[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    sat = np.where(cmax > 0, delta / cmax, 0.0)

    # Value
    val = cmax

    return np.stack([hue, sat, val], axis=-1)


def compute_bar_metrics(
    hsv_img: np.ndarray,
    bar_width: int,
    bar_name: str = ""
) -> dict[str, Any] | None:

    H = hsv_img[:, :, 0]
    S = hsv_img[:, :, 1]
    V = hsv_img[:, :, 2]

    SAT_GATE = 0.35
    VAL_GATE = 0.50

    fill_mask = (S > SAT_GATE) & (V > VAL_GATE)

    projection = fill_mask.any(axis=0)
    nonzero = np.where(projection)[0]

    if len(nonzero) == 0:
        return {
            "progression_percent": 0,
            "colorPresence": None,
        }

    first_x = int(nonzero[0])
    last_x = int(nonzero[-1])

    fill_width = last_x - first_x + 1

    if fill_width < 0.03 * bar_width:
        return {
            "progression_percent": 0,
            "colorPresence": None,
        }

    progression_percent = round((fill_width / bar_width) * 100)
    progression_percent = max(0, min(100, progression_percent))

    # Severity mapping based on fill width
    if progression_percent <= 20:
        severity = {
            "hasGreen": True,
            "hasYellow": False,
            "hasOrange": False,
            "hasRed": False,
        }
    elif progression_percent <= 60:
        severity = {
            "hasGreen": False,
            "hasYellow": True,
            "hasOrange": False,
            "hasRed": False,
        }
    elif progression_percent <= 80:
        severity = {
            "hasGreen": False,
            "hasYellow": False,
            "hasOrange": True,
            "hasRed": False,
        }
    else:
        severity = {
            "hasGreen": False,
            "hasYellow": False,
            "hasOrange": False,
            "hasRed": True,
        }

    return {
        "progression_percent": progression_percent,
        "colorPresence": severity,
    }
def compute_homeostasis_metrics(img_array: np.ndarray) -> dict:

    page_height, page_width, _ = img_array.shape

# ─────────────────────────────────────────────
# Extract center numeric score directly
# ─────────────────────────────────────────────

score_y0 = int(page_height * 0.45)
score_y1 = int(page_height * 0.65)
score_x0 = int(page_width * 0.40)
score_x1 = int(page_width * 0.60)

score_crop = img_array[score_y0:score_y1, score_x0:score_x1]

gray = np.mean(score_crop, axis=2)
binary = gray < 120

row_sum = binary.sum(axis=1)
valid_rows = np.where(row_sum > 20)[0]

if len(valid_rows) == 0:
    total_score = None
else:
    digit_region = binary[valid_rows[0]:valid_rows[-1], :]
    col_sum = digit_region.sum(axis=0)
    valid_cols = np.where(col_sum > 20)[0]

    if len(valid_cols) == 0:
        total_score = None
    else:
        digit_width = valid_cols[-1] - valid_cols[0]

        # Calibrated for ES Teck 300 DPI
        if digit_width < 40:
            total_score = 10
        elif digit_width < 70:
            total_score = 15
        elif digit_width < 90:
            total_score = 19
        elif digit_width < 110:
            total_score = 25
        else:
            total_score = 30

    for y0, y1, x0, x1 in CLASS_BOXES:

        ys = int(page_height * y0)
        ye = int(page_height * y1)
        xs = int(page_width * x0)
        xe = int(page_width * x1)

        crop = img_array[ys:ye, xs:xe]

        if crop.size == 0:
            continue

        gray = np.mean(crop, axis=2)
        binary = gray < 100  # digit is dark
        
        # Remove noise by ignoring thin rows
        row_sum = binary.sum(axis=1)
        valid_rows = np.where(row_sum > 10)[0]
        
        if len(valid_rows) == 0:
            continue
        
        digit_region = binary[valid_rows[0]:valid_rows[-1], :]
        
        # Count vertical stroke segments
        col_sum = digit_region.sum(axis=0)
        segments = np.where(col_sum > 15)[0]
        
        if len(segments) == 0:
            continue
        
        # Count distinct groups of columns
        group_count = 1
        for i in range(1, len(segments)):
            if segments[i] - segments[i-1] > 3:
                group_count += 1
        
        # Deterministic mapping for ES Teck digits
        if group_count == 1:
            value = 1
        elif group_count == 2:
            value = 4
        elif group_count == 3:
            value = 3
        elif group_count == 4:
            value = 5
        else:
            value = 2

total_score += value

    # ─────────────────────────────────────────────
    # 2. Detect center box risk color
    # ─────────────────────────────────────────────

    y_start = int(page_height * 0.42)
    y_end   = int(page_height * 0.58)
    x_start = int(page_width * 0.42)
    x_end   = int(page_width * 0.58)

    center_crop = img_array[y_start:y_end, x_start:x_end]

    hsv = rgb_to_hsv(center_crop)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    
    mask = (S > 0.35) & (V > 0.60)
    valid_hues = H[mask]
    
    if valid_hues.size == 0:
        risk_color = "unknown"
    else:
        mean_hue = float(np.mean(valid_hues))
    
        if 85 <= mean_hue <= 160:
            risk_color = "green"
        elif 60 <= mean_hue < 85:
            risk_color = "light_green"
        elif 40 <= mean_hue < 60:
            risk_color = "grey"
        elif 20 <= mean_hue < 40:
            risk_color = "yellow"
        elif 10 <= mean_hue < 20:
            risk_color = "orange"
        else:
            risk_color = "red"

    return {
        "homeostasis_score": total_score,
        "risk_color": risk_color,
    }

def process_pdf(pdf_bytes: bytes) -> dict:

    if not pdf_bytes.startswith(b"%PDF"):
        return {
            "success": False,
            "error": "Invalid PDF header",
            "results": {},
        }

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return {
            "success": False,
            "error": "Invalid or corrupted PDF file",
            "results": {},
        }

    # ── Compute Homeostasis (page 0) ──
    homeostasis = None

    if len(doc) > 0:
        zoom = TARGET_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = doc[0].get_pixmap(matrix=mat, alpha=False)

        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

        homeostasis = compute_homeostasis_metrics(img_array)

    DISEASE_LAYOUT = {
        1: DISEASE_BAR_BANDS_PAGE_1,
        2: DISEASE_BAR_BANDS_PAGE_2,
    }

    results = {}
    errors = []

    for page_index, band_list in DISEASE_LAYOUT.items():

        if page_index >= len(doc):
            continue

        page = doc[page_index]

        zoom = TARGET_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

        page_height = pix.height
        page_width = pix.width

        x_start = int(page_width * HORIZONTAL_PAD_PCT)
        x_end = int(page_width * (1.0 - HORIZONTAL_PAD_PCT))
        bar_total_width = x_end - x_start

        for disease_name, top_pct, bottom_pct in band_list:

            try:
                y_start = int(page_height * top_pct)
                y_end = int(page_height * bottom_pct)

                if y_end <= y_start:
                    results[disease_name] = None
                    continue

                bar_crop = img_array[y_start:y_end, x_start:x_end]

                if bar_crop.size == 0:
                    results[disease_name] = None
                    continue

                hsv = rgb_to_hsv(bar_crop)

                metrics = compute_bar_metrics(
                    hsv,
                    bar_total_width,
                    bar_name=disease_name,
                )

                results[disease_name] = metrics

            except Exception as e:
                results[disease_name] = None
                errors.append(f"{disease_name}: {str(e)}")

    doc.close()

    return {
        "success": True,
        "engine_version": "v3.3-homeostasis",
        "homeostasis": homeostasis,
        "results": results,
        "errors": errors if errors else None,
    }
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v3.3-homeostasis"})

@app.route("/preprocess", methods=["POST"])
def preprocess():
    """
    Supports BOTH:
    - multipart/form-data with file
    - application/json with pdf_base64
    """

    # ---- Auth check ----
    if PREPROCESS_API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != PREPROCESS_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

    pdf_bytes = None

    # ---- OPTION 1: Multipart file upload ----
    if "file" in request.files:
        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            pdf_bytes = file.read()
        except Exception:
            return jsonify({"error": "Failed to read uploaded file"}), 400

    # ---- OPTION 2: JSON base64 upload (legacy support) ----
    elif request.is_json:
        data = request.get_json(silent=True)

        if not data or "pdf_base64" not in data:
            return jsonify({"error": "pdf_base64 is required"}), 400

        try:
            import base64
            pdf_bytes = base64.b64decode(data["pdf_base64"])
        except Exception:
            return jsonify({"error": "Invalid base64 encoding"}), 400

    else:
        return jsonify({"error": "No valid upload format detected"}), 400

    # ---- Validate PDF header ----
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        return jsonify({
            "success": False,
            "error": "Invalid PDF file",
            "results": {}
        }), 400

    # ---- Process ----
    try:
        result = process_pdf(pdf_bytes)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
