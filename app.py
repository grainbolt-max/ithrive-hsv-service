import os
import re
import fitz
import pytesseract
import tempfile
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"

app = Flask(__name__)


# =========================
# AUTH
# =========================
def is_authorized(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return auth.split(" ")[1] == API_KEY


# =========================
# CLEAN NUMBER
# =========================
def clean_number(val):
    if not val:
        return None
    val = val.replace(",", ".")
    try:
        return float(val)
    except:
        return None


# =========================
# STRONG OCR PARSER
# =========================
def extract_hrv_from_text(text):
    text = text.replace(",", ".")
    
    rmssd_match = re.search(r"R\s*M\s*S\s*S\s*D.*?(\d+\.\d+|\d+)", text, re.IGNORECASE)
    lf_hf_match = re.search(r"L\s*F\s*[/\s]\s*H\s*F.*?(\d+\.\d+|\d+)", text, re.IGNORECASE)
    total_power_match = re.search(r"Total\s*Power.*?(\d+\.\d+|\d+)", text, re.IGNORECASE)

    rmssd = clean_number(rmssd_match.group(1)) if rmssd_match else None
    lf_hf = clean_number(lf_hf_match.group(1)) if lf_hf_match else None
    total_power = clean_number(total_power_match.group(1)) if total_power_match else None

    if rmssd is None or lf_hf is None:
        return None

    return {
        "rmssd_ms": rmssd,
        "lf_hf_ratio": lf_hf,
        "total_power_ms2": total_power
    }


# =========================
# SAFE LOW MEMORY OCR
# =========================
def ocr_page_low_memory(page):
    mat = fitz.Matrix(0.6, 0.6)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pix.save(tmp.name)
        path = tmp.name

    text = pytesseract.image_to_string(path)
    os.remove(path)

    return text


# =========================
# MAIN ENDPOINT
# =========================
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

        hrv_data = None

        for page in doc:
            if "hrv" not in page.get_text().lower():
                continue

            text = ocr_page_low_memory(page)
            hrv_data = extract_hrv_from_text(text)

            if hrv_data:
                break

        doc.close()

        if not hrv_data:
            return jsonify({"error": "hrv_not_detected"}), 422

        return jsonify(hrv_data)

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "minimal_hrv_service_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
