import re
import fitz
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"
app = Flask(__name__)


# =========================
# AUTH
# =========================
def is_authorized(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"


# =========================
# CLEAN NUMBER
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
# PARSE HRV FROM TEXT
# =========================
def parse_hrv(text):

    text = text.replace(",", ".")

    def find(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return clean_number(match.group(1)) if match else None

    return {
        "k30_15_ratio": find(r"K30/15.*?Value:\s*(\d+\.\d+|\d+)"),
        "valsava_ratio": find(r"Valsalva ratio.*?Value:\s*(\d+\.\d+|\d+)"),
        "lf_hf_ratio": find(r"LF/HF.*?(\d+\.\d+|\d+)"),
        "total_power": find(r"Total Power.*?(\d+\.\d+|\d+)"),
    }


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


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "text_only_hrv_service_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
