import re
import fitz
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"
app = Flask(__name__)


# =========================
# AUTH CHECK
# =========================
def is_authorized(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"


# =========================
# SAFE FLOAT CLEANER
# =========================
def clean_number(value):
    if not value:
        return None
    value = value.replace(",", ".")
    try:
        return float(value)
    except:
        return None


# =========================
# HRV PARSER
# =========================
def parse_hrv(text):

    text = text.replace(",", ".")

    def find_after(label):
        pattern = rf"{label}[\s\S]*?Value:\s*(\d+\.?\d*)"
        match = re.search(pattern, text, re.IGNORECASE)
        return clean_number(match.group(1)) if match else None

    return {
        "k30_15_ratio": find_after("K30/15"),
        "valsava_ratio": find_after("Valsalva ratio"),
        "lf_hf_ratio": find_after("LF/HF"),
        "total_power": find_after("Total Power"),
    }


# =========================
# MAIN EXTRACTION ENDPOINT
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
        combined_text = ""

        for page in doc:
            combined_text += page.get_text()

        doc.close()

        parsed = parse_hrv(combined_text)

        if not parsed["k30_15_ratio"]:
            return jsonify({"error": "hrv_not_detected"}), 422

        return jsonify(parsed)

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


# =========================
# HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "text_only_hrv_service_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
