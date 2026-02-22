import io
import re
import pdfplumber
from flask import Flask, request, jsonify

app = Flask(__name__)

AUTH_TOKEN = "ithrive_secure_2026_key"


def extract_float(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


def extract_report(pdf_bytes):
    result = {
        "body_composition": None,
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text()

            if not text:
                continue

            # ---------------------------------------------------
            # BODY COMPOSITION DEBUG PRINT
            # ---------------------------------------------------
            if "Body composition and follow up" in text:
                print("\n==============================")
                print("BODY PAGE DETECTED")
                print("Page index:", page_index)
                print("----- BODY PAGE TEXT START -----")
                print(text)
                print("----- BODY PAGE TEXT END -----")
                print("==============================\n")

            # ---------------------------------------------------
            # HRV
            # ---------------------------------------------------
            k_match = re.search(r"K30\/15.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
            if k_match:
                result["hrv"]["k30_15_ratio"] = float(k_match.group(1))

            v_match = re.search(r"Valsalva.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
            if v_match:
                result["hrv"]["valsava_ratio"] = float(v_match.group(1))

            # ---------------------------------------------------
            # Daily Energy Expenditure
            # ---------------------------------------------------
            dee_match = re.search(r"Daily Energy Expenditure.*?([0-9]+)\s*Kcal", text)
            if dee_match:
                result["metabolic"]["daily_energy_expenditure_kcal"] = float(dee_match.group(1))

            # ---------------------------------------------------
            # Blood Pressure
            # ---------------------------------------------------
            bp_match = re.search(
                r"Systolic\s*\/\s*Diastolic\s*pressure:\s*([0-9]+)\s*\/\s*([0-9]+)",
                text
            )
            if bp_match:
                result["vitals"]["systolic_bp"] = float(bp_match.group(1))
                result["vitals"]["diastolic_bp"] = float(bp_match.group(2))

    return result


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "BODY_TEXT_DEBUG_ACTIVE"})


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
