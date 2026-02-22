import io
import re
import pdfplumber
from flask import Flask, request, jsonify

app = Flask(__name__)

AUTH_TOKEN = "ithrive_secure_2026_key"


def safe_float(value):
    try:
        return float(value)
    except:
        return None


def extract_report(pdf_bytes):
    result = {
        "body_composition": {},
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
            # BODY COMPOSITION REGEX EXTRACTION
            # ---------------------------------------------------
            if "Body composition and follow up" in text:

                # DEBUG (prints full flattened page once)
                print("\n==============================")
                print("BODY PAGE DETECTED")
                print("Page index:", page_index)
                print("----- FULL PAGE TEXT START -----")
                print(text)
                print("----- FULL PAGE TEXT END -----")
                print("==============================\n")

                body = {}

                body["weight_kg"] = safe_float(
                    re.search(r"Weight.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE).group(1)
                ) if re.search(r"Weight.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE) else None

                body["fat_mass_kg"] = safe_float(
                    re.search(r"Fat\s*mass.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE).group(1)
                ) if re.search(r"Fat\s*mass.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE) else None

                body["fat_free_mass_kg"] = safe_float(
                    re.search(r"Fat\s*free\s*mass.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE).group(1)
                ) if re.search(r"Fat\s*free\s*mass.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE) else None

                body["skeletal_muscle_mass_kg"] = safe_float(
                    re.search(r"Skeletal\s*muscle\s*mass.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE).group(1)
                ) if re.search(r"Skeletal\s*muscle\s*mass.*?([0-9]+\.?[0-9]*)\s*kg", text, re.IGNORECASE) else None

                body["body_fat_percent"] = safe_float(
                    re.search(r"Body\s*fat.*?([0-9]+\.?[0-9]*)\s*%", text, re.IGNORECASE).group(1)
                ) if re.search(r"Body\s*fat.*?([0-9]+\.?[0-9]*)\s*%", text, re.IGNORECASE) else None

                result["body_composition"] = body

            # ---------------------------------------------------
            # HRV
            # ---------------------------------------------------
            k_match = re.search(r"K30\/15.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
            if k_match:
                result["hrv"]["k30_15_ratio"] = safe_float(k_match.group(1))

            v_match = re.search(r"Valsalva.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
            if v_match:
                result["hrv"]["valsava_ratio"] = safe_float(v_match.group(1))

            # ---------------------------------------------------
            # Daily Energy Expenditure
            # ---------------------------------------------------
            dee_match = re.search(r"Daily Energy Expenditure.*?([0-9]+)\s*Kcal", text)
            if dee_match:
                result["metabolic"]["daily_energy_expenditure_kcal"] = safe_float(dee_match.group(1))

            # ---------------------------------------------------
            # Blood Pressure
            # ---------------------------------------------------
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
    return jsonify({"status": "PRODUCTION_REGEX_VERSION_ACTIVE"})


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
