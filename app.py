import io
import re
import pdfplumber
from flask import Flask, request, jsonify

app = Flask(__name__)

AUTH_TOKEN = "ithrive_secure_2026_key"


def safe_float(val):
    try:
        return float(val)
    except:
        return None


def extract_body_from_words(words):
    body = {}

    for i, word in enumerate(words):
        text = word["text"]

        if text.lower() == "weight":
            # Look ahead for first numeric value
            for j in range(i, min(i+10, len(words))):
                if re.match(r"^\d+\.?\d*$", words[j]["text"]):
                    body["weight_lb"] = safe_float(words[j]["text"])
                    break

        if text.lower() == "mass":
            prev = words[i-1]["text"].lower() if i > 0 else ""
            if prev == "fat":
                for j in range(i, min(i+10, len(words))):
                    if re.match(r"^\d+\.?\d*$", words[j]["text"]):
                        body["fat_mass_lb"] = safe_float(words[j]["text"])
                        break

        if text.lower() == "water":
            prev = words[i-1]["text"].lower() if i > 0 else ""
            if prev == "total":
                for j in range(i, min(i+10, len(words))):
                    if re.match(r"^\d+\.?\d*$", words[j]["text"]):
                        body["total_body_water_lb"] = safe_float(words[j]["text"])
                        break

    return body


def extract_report(pdf_bytes):
    result = {
        "body_composition": {},
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:

            text = page.extract_text()
            words = page.extract_words()

            if text and "Body Composition Indicators" in text:
                body_data = extract_body_from_words(words)
                result["body_composition"] = body_data

            if text:
                # HRV
                k_match = re.search(r"K30\/15.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
                if k_match:
                    result["hrv"]["k30_15_ratio"] = safe_float(k_match.group(1))

                v_match = re.search(r"Valsalva.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
                if v_match:
                    result["hrv"]["valsava_ratio"] = safe_float(v_match.group(1))

                # Energy
                dee_match = re.search(r"Daily Energy Expenditure.*?([0-9]+)", text)
                if dee_match:
                    result["metabolic"]["daily_energy_expenditure_kcal"] = safe_float(dee_match.group(1))

                # Blood Pressure
                bp_match = re.search(
                    r"Systolic\s*\/\s*Diastolic\s*pressure:\s*([0-9]+)\s*\/\s*([0-9]+)",
                    text
                )
                if bp_match:
                    result["vitals"]["systolic_bp"] = safe_float(bp_match.group(1))
                    result["vitals"]["diastolic_bp"] = safe_float(bp_match.group(2))

    return result


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "COORDINATE_EXTRACTION_ACTIVE"})


@app.route("/v1/extract-report", methods=["POST"])
def extract_report_endpoint():
    auth_header = request.headers.get("Authorization")

    if auth_header != f"Bearer {AUTH_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    try:
        result = extract_report(pdf_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
