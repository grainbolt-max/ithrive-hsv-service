import os
import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2

from flask import Flask, request, jsonify

app = Flask(__name__)

PREPROCESS_API_KEY = os.environ.get("PREPROCESS_API_KEY")

TARGET_DPI = 150


# ============================================================
# Utilities
# ============================================================

def render_page_to_image(page):
    zoom = TARGET_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def extract_text_with_ocr(page):
    img = render_page_to_image(page)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text


# ============================================================
# HRV Extraction (Deterministic)
# ============================================================

def extract_hrv_metrics(pdf_bytes: bytes) -> dict:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return {"rmssd_ms": None, "lf_hf_ratio": None}

    rmssd_value = None
    lf_hf_value = None

    for page in doc:
        text = page.get_text()

        # OCR fallback if text layer empty
        if not text.strip():
            text = extract_text_with_ocr(page)

        lines = text.split("\n")

        for line in lines:
            # Match RMSSD
            if re.search(r"\bRMSSD\b", line, re.IGNORECASE):
                match = re.search(r"(\d+\.?\d*)", line)
                if match:
                    rmssd_value = float(match.group(1))

            # Match LF/HF
            if re.search(r"\bLF[/\- ]?HF\b", line, re.IGNORECASE):
                match = re.search(r"(\d+\.?\d*)", line)
                if match:
                    lf_hf_value = float(match.group(1))

    return {
        "rmssd_ms": rmssd_value,
        "lf_hf_ratio": lf_hf_value
    }


# ============================================================
# Debug Endpoint (TEMPORARY)
# ============================================================

@app.route("/debug-text", methods=["POST"])
def debug_text():
    if PREPROCESS_API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != PREPROCESS_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return jsonify({"error": "Invalid PDF"}), 400

    full_text = ""

    for page in doc:
        text = page.get_text()

        if not text.strip():
            text = extract_text_with_ocr(page)

        full_text += "\n\n===== PAGE =====\n\n"
        full_text += text

    return jsonify({
        "extracted_text": full_text
    })


# ============================================================
# Extract HRV Endpoint
# ============================================================

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
    return jsonify(result)


# ============================================================
# Preprocess Endpoint (Homeostasis placeholder)
# ============================================================

@app.route("/preprocess", methods=["POST"])
def preprocess():
    if PREPROCESS_API_KEY:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != PREPROCESS_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    return jsonify({
        "engine_version": "v6.0-deterministic-hrv",
        "homeostasis": {
            "homeostasis_score": None,
            "risk_color": "unknown"
        },
        "success": True
    })


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "v6.0-deterministic-hrv"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
