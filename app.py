import re
import fitz
import pytesseract
from PIL import Image
import io
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"
app = Flask(__name__)


def is_authorized(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"


def clean_number(value):
    if not value:
        return None
    value = value.replace(",", ".")
    try:
        return float(value)
    except:
        return None


def ocr_page(page):
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    return text


def parse_hrv(text):
    text = text.replace(",", ".")

    def find(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return clean_number(match.group(1)) if match else None

    return {
        "rmssd_ms": find(r"RMSSD.*?(\d+\.\d+|\d+)"),
        "sdnn_ms": find(r"SDNN.*?(\d+\.\d+|\d+)"),
        "hr_bpm": find(r"Heart rate.*?(\d+\.\d+|\d+)"),
        "lf_percent": find(r"Power\s*LF.*?(\d+\.\d+|\d+)"),
        "hf_percent": find(r"Power\s*HF.*?(\d+\.\d+|\d+)"),
        "vlf_percent": find(r"Power\s*VLF.*?(\d+\.\d+|\d+)"),
        "lf_hf_ratio": find(r"LF\s*/\s*HF.*?(\d+\.\d+|\d+)"),
        "total_power_ms2": find(r"Total\s*Power.*?(\d+\.\d+|\d+)")
    }


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

        combined_text = ""

        for page in doc:
            combined_text += ocr_page(page)

        doc.close()

        parsed = parse_hrv(combined_text)

        if not parsed["rmssd_ms"]:
            return jsonify({"error": "hrv_not_detected"}), 422

        return jsonify(parsed)

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ocr_hrv_service_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
