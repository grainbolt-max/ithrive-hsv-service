# ENGINE POLICY:
# The uploaded ES Teck PDF must NOT include the bottom color legend.
# Bars are assumed to be pre-cleaned before upload.
# Engine will not attempt legend suppression.

"""
ES Teck Bio Scan — Deterministic Disease Bar Preprocessing Service
Version: v3.0-stable-precleaned-input

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
TARGET_DPI = 400
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
DISEASE_BAR_BANDS = [
    ("atherosclerosis", 0.155, 0.185),
    ("lv_hypertrophy", 0.190, 0.220),
    ("large_artery_stiffness", 0.225, 0.255),
    ("small_medium_artery_stiffness", 0.260, 0.290),
    ("peripheral_vessels", 0.295, 0.325),
    ("diabetes_screening", 0.330, 0.360),
    ("insulin_resistance", 0.365, 0.395),
    ("metabolic_syndrome", 0.400, 0.430),
    ("ldl_cholesterol", 0.435, 0.465),
    ("chronic_hepatitis", 0.470, 0.500),
    ("hepatic_fibrosis", 0.505, 0.535),
    ("kidney_function", 0.540, 0.570),
    ("digestive_disorders", 0.575, 0.605),
    ("respiratory", 0.610, 0.640),
    ("hyperthyroidism", 0.645, 0.675),
    ("hypothyroidism", 0.680, 0.710),
    ("major_depression", 0.715, 0.745),
    ("tissue_inflammatory_process", 0.750, 0.780),
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

    SAT_GATE = 0.70
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

    # RIGHT EDGE hue sampling
    EDGE_WIDTH = 3
    edge_start = max(first_x, last_x - EDGE_WIDTH + 1)
    edge_end = last_x + 1

    edge_h = H[:, edge_start:edge_end]
    edge_s = S[:, edge_start:edge_end]
    edge_v = V[:, edge_start:edge_end]

    valid_mask = (edge_s > SAT_GATE) & (edge_v > VAL_GATE)
    valid_hues = edge_h[valid_mask]

    if valid_hues.size == 0:
        return {
            "progression_percent": progression_percent,
            "colorPresence": None,
        }

    edge_hue = float(np.median(valid_hues))

    hasGreen  = 70 <= edge_hue <= 160
    hasYellow = 20 <= edge_hue < 70
    hasOrange = False
    hasRed    = edge_hue < 20 or edge_hue > 160
    

    return {
        "progression_percent": progression_percent,
        "colorPresence": {
            "hasGreen": hasGreen,
            "hasYellow": hasYellow,
            "hasOrange": hasOrange,
            "hasRed": hasRed,
        },
    }
def process_pdf(pdf_bytes: bytes) -> dict:

    # Fast header validation (deterministic guard)
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

    DS_PAGE_INDEX = 1

    if len(doc) <= DS_PAGE_INDEX:
        doc.close()
        return {
            "success": False,
            "error": f"PDF has only {len(doc)} pages; expected >= {DS_PAGE_INDEX + 1}",
            "results": {},
        }

    page = doc[DS_PAGE_INDEX]

    zoom = TARGET_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    actual_dpi_x = pix.width / (page.rect.width / 72.0)

    if actual_dpi_x < MIN_DPI:
        doc.close()
        return {
            "success": False,
            "error": f"Rendered resolution {actual_dpi_x:.0f} DPI < minimum {MIN_DPI} DPI",
            "results": {},
        }

    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )

    page_height = pix.height
    page_width = pix.width

    x_start = int(page_width * HORIZONTAL_PAD_PCT)
    x_end = int(page_width * (1.0 - HORIZONTAL_PAD_PCT))
    bar_total_width = x_end - x_start

    results = {}
    errors = []

    for disease_name, top_pct, bottom_pct in DISEASE_BAR_BANDS:
        try:
            y_start = int(page_height * top_pct)
            y_end = int(page_height * bottom_pct)

            if y_end <= y_start or y_end > page_height:
                results[disease_name] = None
                errors.append(f"{disease_name}: invalid crop region")
                continue

            bar_crop = img_array[y_start:y_end, x_start:x_end]

            if bar_crop.size == 0:
                results[disease_name] = None
                errors.append(f"{disease_name}: empty crop")
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
        "engine_version": "v3.0-stable-precleaned-input",
        "page_index": DS_PAGE_INDEX,
        "resolution_dpi": round(actual_dpi_x),
        "results": results,
        "errors": errors if errors else None,
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v3.0-stable-precleaned-input"})

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
