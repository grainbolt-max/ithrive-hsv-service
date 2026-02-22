from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import re
import os

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "ithrive_secure_2026_key")


def require_auth():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ")[1]
    return token == API_KEY


def extract_text_from_pdf(file_storage):
    doc = fitz.open(stream=file_storage.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


def parse_float(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


def parse_string(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "extract_report_service_running"})


@app.route("/extract-hrv", methods=["POST"])
def extract_hrv():
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "file_missing"}), 400

    file = request.files["file"]
    text = extract_text_from_pdf(file)

    result = {
        "k30_15_ratio": parse_float(r"K30\/15.*?Value:\s*([0-9.]+)", text),
        "valsava_ratio": parse_float(r"Valsalva ratio.*?Value:\s*([0-9.]+)", text),
        "systolic_bp": parse_float(r"Systolic\s*\/\s*Diastolic pressure:\s*([0-9.]+)", text),
        "diastolic_bp": parse_float(r"Systolic\s*\/\s*Diastolic pressure:\s*[0-9.]+\s*\/\s*([0-9.]+)", text),
        "daily_energy_expenditure_kcal": parse_float(r"Daily Energy Expenditure.*?:\s*([0-9.]+)", text),
    }

    if all(value is None for value in result.values()):
        return jsonify({"error": "hrv_not_detected"}), 422

    return jsonify(result)


@app.route("/v1/extract-report", methods=["POST"])
def extract_report():
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "file_missing"}), 400

    file = request.files["file"]
    text = extract_text_from_pdf(file)

    response = {
        "patient": {
            "first_name": parse_string(r"First\/Last Name:\s*([A-Za-z]+)", text),
            "last_name": parse_string(r"First\/Last Name:\s*[A-Za-z]+\s+([A-Za-z]+)", text),
            "dob": parse_string(r"Date of birth:\s*([0-9\-\/]+)", text),
            "gender": parse_string(r"Gender:\s*(Male|Female)", text),
            "exam_date": parse_string(r"Examination performed at:\s*([0-9\-\s:]+)", text),
        },
        "vitals": {
            "systolic_bp": parse_float(r"Systolic\s*\/\s*Diastolic pressure:\s*([0-9.]+)", text),
            "diastolic_bp": parse_float(r"Systolic\s*\/\s*Diastolic pressure:\s*[0-9.]+\s*\/\s*([0-9.]+)", text),
        },
        "hrv": {
            "k30_15_ratio": parse_float(r"K30\/15.*?Value:\s*([0-9.]+)", text),
            "valsava_ratio": parse_float(r"Valsalva ratio.*?Value:\s*([0-9.]+)", text),
        },
        "metabolic": {
            "daily_energy_expenditure_kcal": parse_float(r"Daily Energy Expenditure.*?:\s*([0-9.]+)", text),
        },
        "body": {
            "weight_lbs": parse_float(r"Weight\s*:\s*([0-9.]+)", text),
            "height_feet": parse_float(r"Height:\s*([0-9.]+)\s*Feet", text),
            "height_inches": parse_float(r"Height:\s*[0-9.]+\s*Feet\s*([0-9.]+)\s*Inch", text),
        },
    }

    # strict validation
    flat_values = []

    for section in response.values():
        for value in section.values():
            flat_values.append(value)

    if all(value is None for value in flat_values):
        return jsonify({"error": "Report format not supported"}), 422

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
