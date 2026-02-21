import re
import fitz
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


def extract_value_after_label(text, label):
    pattern = rf"{label}[\s\S]*?Value:\s*(\d+\.?\d*)"
    match = re.search(pattern, text, re.IGNORECASE)
    return clean_number(match.group(1)) if match else None


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

        # HRV values present in YOUR report format
        k30_15 = extract_value_after_label(combined_text, "K30/15")
        valsalva = extract_value_after_label(combined_text, "Valsalva ratio")

        # Blood pressure
        bp_match = re.search(r"Systolic\s*/\s*Diastolic pressure:\s*(\d+)\s*/\s*(\d+)", combined_text)
        systolic = clean_number(bp_match.group(1)) if bp_match else None
        diastolic = clean_number(bp_match.group(2)) if bp_match else None

        # Daily Energy Expenditure
        dee_match = re.search(r"Daily Energy Expenditure \(DEE\):\s*(\d+)", combined_text)
        dee = clean_number(dee_match.group(1)) if dee_match else None

        result = {
            "k30_15_ratio": k30_15,
            "valsava_ratio": valsalva,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "daily_energy_expenditure_kcal": dee
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "stable_text_extraction_service_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
