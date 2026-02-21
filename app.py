import re
import fitz
import pytesseract
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"

app = Flask(__name__)


# -------------------------
# Authorization
# -------------------------
def is_authorized(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth.split(" ")[1]
    return token == API_KEY


# -------------------------
# Extract HRV values
# -------------------------
def extract_hrv_from_text(text):
    rmssd_match = re.search(r"RMSSD\s*[:\-]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    lf_hf_match = re.search(r"LF\s*/\s*HF.*?(\d+\.?\d*)", text, re.IGNORECASE)
    total_power_match = re.search(r"Total\s*power\s*[:\-]?\s*(\d+\.?\d*)", text, re.IGNORECASE)

    rmssd = float(rmssd_match.group(1)) if rmssd_match else None
    lf_hf = float(lf_hf_match.group(1)) if lf_hf_match else None
    total_power = float(total_power_match.group(1)) if total_power_match else None

    if rmssd is None or lf_hf is None:
        return None

    return {
        "rmssd_ms": rmssd,
        "lf_hf_ratio": lf_hf,
        "total_power_ms2": total_power
    }


# -------------------------
# HRV Endpoint
# -------------------------
@app.route("/extract-hrv", methods=["POST"])
def extract_hrv():
    if not is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "file_missing"}), 400

    try:
        file = request.files["file"]
        pdf_bytes = file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        hrv_values = None

        # Process one page at a time
        for page in doc:

            # Only OCR pages likely containing HRV
            text_layer = page.get_text().lower()

            if "hrv" not in text_layer:
                continue

            # Low DPI to protect memory
            pix = page.get_pixmap(dpi=100)

            ocr_text = pytesseract.image_to_string(pix.tobytes("png"))

            hrv_values = extract_hrv_from_text(ocr_text)

            if hrv_values:
                break

        doc.close()

        if not hrv_values:
            return jsonify({
                "error": "hrv_not_detected"
            }), 422

        return jsonify(hrv_values)

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


# -------------------------
# Health Check
# -------------------------
@app.route("/")
def health():
    return jsonify({"status": "hrv_ocr_service_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
