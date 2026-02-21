import re
import os
import fitz  # PyMuPDF
import pytesseract
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"

app = Flask(__name__)

# ---- Authorization ----
def check_auth():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ")[1]
    return token == API_KEY


# ---- Extract HRV from OCR Text ----
def extract_hrv_values(text):
    rmssd_match = re.search(r"RMSSD\s*[:\-]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    lf_hf_match = re.search(r"(LF\s*/\s*HF|Ratio of ANS activity).*?(\d+\.?\d*)", text, re.IGNORECASE)
    total_power_match = re.search(r"Total power\s*[:\-]?\s*(\d+\.?\d*)", text, re.IGNORECASE)

    return {
        "rmssd_ms": float(rmssd_match.group(1)) if rmssd_match else None,
        "lf_hf_ratio": float(lf_hf_match.group(2)) if lf_hf_match else None,
        "total_power_ms2": float(total_power_match.group(1)) if total_power_match else None
    }


# ---- Main HRV Endpoint ----
@app.route("/extract-hrv", methods=["POST"])
def extract_hrv():
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")

        combined_text = ""

        # Process one page at a time at LOW DPI
        for page in doc:
            pix = page.get_pixmap(dpi=120)  # Low DPI to reduce memory
            text = pytesseract.image_to_string(pix.tobytes("png"))
            combined_text += text + "\n"

            # Early exit if we already found RMSSD
            if "RMSSD" in combined_text:
                break

        doc.close()

        values = extract_hrv_values(combined_text)

        return jsonify(values)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Health Check ----
@app.route("/")
def home():
    return jsonify({"status": "HRV OCR extractor running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
