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
# TEXT PARSER
# =========================
def parse_report_text(text):

    text = text.replace(",", ".")

    def find(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return clean_number(match.group(1)) if match else None

    data = {
        "patient": {
            "name": re.search(r"Patient:\s*(.+)", text).group(1).strip() if re.search(r"Patient:\s*(.+)", text) else None,
            "age": find(r"Age:\s*(\d+)"),
            "gender": re.search(r"Gender:\s*(\w+)", text).group(1) if re.search(r"Gender:\s*(\w+)", text) else None,
        },
        "time_domain": {
            "rmssd_ms": find(r"RMSSD.*?(\d+\.\d+|\d+)"),
            "sdnn_ms": find(r"SDNN.*?(\d+\.\d+|\d+)"),
            "hr_bpm": find(r"Heart rate.*?(\d+\.\d+|\d+)"),
            "mean_rr_ms": find(r"Mean value of RR intervals.*?(\d+\.\d+|\d+)"),
            "nn50": find(r"NN50.*?(\d+\.\d+|\d+)"),
            "pnn50_percent": find(r"pNN50.*?(\d+\.\d+|\d+)"),
            "stress_index": find(r"Stress index.*?(\d+\.\d+|\d+)"),
        },
        "frequency_domain": {
            "lf_percent": find(r"Power LF.*?(\d+\.\d+|\d+)"),
            "hf_percent": find(r"Power HF.*?(\d+\.\d+|\d+)"),
            "vlf_percent": find(r"Power VLF.*?(\d+\.\d+|\d+)"),
            "lf_hf_ratio": find(r"LF\s*/\s*HF.*?(\d+\.\d+|\d+)"),
            "total_power_ms2": find(r"Total power.*?(\d+\.\d+|\d+)"),
        }
    }

    return data


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

        full_text = ""
        for page in doc:
            full_text += page.get_text()

        doc.close()

        if not full_text.strip():
            return jsonify({"error": "no_text_found"}), 422

        parsed = parse_report_text(full_text)

        return jsonify(parsed)

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "hrv_service_running_no_ocr"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
