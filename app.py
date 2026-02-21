import os
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

PREPROCESS_API_KEY = os.environ.get("PREPROCESS_API_KEY", "")

TARGET_DPI = 300
HORIZONTAL_PAD_PCT = 0.05


# ─────────────────────────────────────────────
# RGB → HSV (deterministic)
# ─────────────────────────────────────────────
def rgb_to_hsv(image_array: np.ndarray) -> np.ndarray:
    rgb = image_array.astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)

    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)

    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    sat = np.where(cmax > 0, delta / cmax, 0.0)
    val = cmax

    return np.stack([hue, sat, val], axis=-1)


# ─────────────────────────────────────────────
# Extract Homeostasis Score (CENTER ONLY)
# ─────────────────────────────────────────────
def compute_homeostasis_metrics(img_array: np.ndarray) -> dict:
    page_height, page_width, _ = img_array.shape

    score_y0 = int(page_height * 0.46)
    score_y1 = int(page_height * 0.64)
    score_x0 = int(page_width * 0.42)
    score_x1 = int(page_width * 0.58)

    score_crop = img_array[score_y0:score_y1, score_x0:score_x1]

    gray = np.mean(score_crop, axis=2)
    binary = gray < 120

    row_sum = binary.sum(axis=1)
    valid_rows = np.where(row_sum > 25)[0]

    total_score = None

    if len(valid_rows) > 0:
        digit_region = binary[valid_rows[0]:valid_rows[-1], :]
        col_sum = digit_region.sum(axis=0)
        valid_cols = np.where(col_sum > 25)[0]

        if len(valid_cols) > 0:
            digit_width = valid_cols[-1] - valid_cols[0]

            # Deterministic mapping for ES Teck 300 DPI
            if digit_width < 70:
                total_score = 10
            elif digit_width < 95:
                total_score = 19
            elif digit_width < 120:
                total_score = 25
            else:
                total_score = 30

    # Detect risk color
    y_start = int(page_height * 0.42)
    y_end = int(page_height * 0.58)
    x_start = int(page_width * 0.42)
    x_end = int(page_width * 0.58)

    center_crop = img_array[y_start:y_end, x_start:x_end]
    hsv = rgb_to_hsv(center_crop)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]

    mask = S > 0.20
    valid_hues = H[mask]

    risk_color = "unknown"

    if valid_hues.size > 0:
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


# ─────────────────────────────────────────────
# Process PDF
# ─────────────────────────────────────────────
def process_pdf(pdf_bytes: bytes) -> dict:
    if not pdf_bytes.startswith(b"%PDF"):
        return {"success": False, "error": "Invalid PDF file", "results": {}}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return {"success": False, "error": "Corrupted PDF", "results": {}}

    homeostasis = None

    if len(doc) > 0:
        zoom = TARGET_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = doc[0].get_pixmap(matrix=mat, alpha=False)

        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

        homeostasis = compute_homeostasis_metrics(img_array)

    doc.close()

    return {
        "success": True,
        "engine_version": "v4.0-center-score",
        "homeostasis": homeostasis,
        "results": {},
        "errors": None,
    }


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v4.0-center-score"})


@app.route("/preprocess", methods=["POST"])
def preprocess():
    if PREPROCESS_API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != PREPROCESS_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    result = process_pdf(pdf_bytes)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
