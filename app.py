import os
import re
import fitz
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

PREPROCESS_API_KEY = os.environ.get("PREPROCESS_API_KEY", "")
TARGET_DPI = 300


# ─────────────────────────────────────────────
# RGB → HSV (for risk color only)
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
# Extract Homeostasis Score via TEXT
# ─────────────────────────────────────────────
def extract_homeostasis_score(page) -> int | None:
    text = page.get_text()
    match = re.search(r"Homeostasis\s*Score\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# ─────────────────────────────────────────────
# Detect Risk Color (pixel-based)
# ─────────────────────────────────────────────
def detect_risk_color(page) -> str:
    zoom = TARGET_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )

    page_height, page_width, _ = img_array.shape

    y0 = int(page_height * 0.48)
    y1 = int(page_height * 0.60)
    x0 = int(page_width * 0.40)
    x1 = int(page_width * 0.60)

    crop = img_array[y0:y1, x0:x1]

    hsv = rgb_to_hsv(crop)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]

    mask = S > 0.15
    valid_hues = H[mask]

    if valid_hues.size == 0:
        return "unknown"

    mean_hue = float(np.mean(valid_hues))

    if 85 <= mean_hue <= 160:
        return "green"
    elif 60 <= mean_hue < 85:
        return "light_green"
    elif 40 <= mean_hue < 60:
        return "grey"
    elif 20 <= mean_hue < 40:
        return "yellow"
    elif 10 <= mean_hue < 20:
        return "orange"
    else:
        return "red"


# ─────────────────────────────────────────────
# Extract HRV (Deterministic Text-Based)
# ─────────────────────────────────────────────
def extract_hrv_metrics(pdf_bytes: bytes) -> dict:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return {"rmssd_ms": None, "lf_hf_ratio": None}

    rmssd_value = None
    lfhf_value = None

    for page in doc:
        text = page.get_text()

        if "RMSSD" not in text and "LF" not in text:
            continue

        lines = text.split("\n")

        for line in lines:
            # Match RMSSD row
            if re.search(r"\bRMSSD\b", line, re.IGNORECASE):
                match = re.search(r"(\d+\.\d+|\d+)", line)
                if match:
                    rmssd_value = float(match.group(1))

            # Match LF/HF row
            if re.search(r"LF\s*/\s*HF", line, re.IGNORECASE) or \
               re.search(r"Ratio of ANS activity", line, re.IGNORECASE):

                match = re.search(r"(\d+\.\d+|\d+)", line)
                if match:
                    lfhf_value = float(match.group(1))

        if rmssd_value is not None and lfhf_value is not None:
            break

    doc.close()

    return {
        "rmssd_ms": rmssd_value,
        "lf_hf_ratio": lfhf_value,
    }


# ─────────────────────────────────────────────
# Process PDF (Homeostasis Only)
# ─────────────────────────────────────────────
def process_pdf(pdf_bytes: bytes) -> dict:
    if not pdf_bytes.startswith(b"%PDF"):
        return {"success": False, "error": "Invalid PDF file"}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return {"success": False, "error": "Corrupted PDF"}

    homeostasis_score = None
    risk_color = "unknown"

    if len(doc) > 0:
        page = doc[0]
        homeostasis_score = extract_homeostasis_score(page)
        risk_color = detect_risk_color(page)

    doc.close()

    return {
        "success": True,
        "engine_version": "v6.0-deterministic-hrv",
        "homeostasis": {
            "homeostasis_score": homeostasis_score,
            "risk_color": risk_color,
        },
    }


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "v6.0-deterministic-hrv"})


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


@app.route("/extract-hrv", methods=["POST"])
def extract_hrv():
    if PREPROCESS_API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != PREPROCESS_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    result = extract_hrv_metrics(pdf_bytes)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
