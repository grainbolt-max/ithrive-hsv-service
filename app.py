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


@app.route("/v1/extract-report", methods=["POST"])
def extract_report():

    if not is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "file_missing"}), 400

    try:
        file = request.files["file"]
        pdf_bytes = file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()

        # -----------------------------
        # PATIENT
        # -----------------------------

        name_match = re.search(r"First/Last Name:\s*(.+)", text)
        name = name_match.group(1).strip() if name_match else None

        first_name = None
        last_name = None
        if name:
            parts = name.split(" ")
            first_name = parts[0]
            last_name = " ".join(parts[1:]) if len(parts) > 1 else None

        dob_match = re.search(r"Date of birth:\s*(.+)", text)
        dob = dob_match.group(1).strip() if dob_match else None

        gender_match = re.search(r"\n(Female|Male)\n", text)
        gender = gender_match.group(1) if gender_match else None

        exam_match = re.search(r"Examination performed at:\s*(.+)", text)
        exam_date = exam_match.group(1).strip() if exam_match else None

        # -----------------------------
        # VITALS
        # -----------------------------

        bp_match = re.search(
            r"Systolic\s*/\s*Diastolic pressure:\s*(\d+)\s*/\s*(\d+)",
            text
        )

        systolic = clean_number(bp_match.group(1)) if bp_match else None
        diastolic = clean_number(bp_match.group(2)) if bp_match else None

        # -----------------------------
        # HRV
        # -----------------------------

        k30_15 = extract_value_after_label(text, "K30/15")
        valsalva = extract_value_after_label(text, "Valsalva ratio")

        # -----------------------------
        # METABOLIC
        # -----------------------------

        dee_match = re.search(
            r"Daily Energy Expenditure \(DEE\):\s*(\d+)",
            text
        )
        dee = clean_number(dee_match.group(1)) if dee_match else None

        # -----------------------------
        # BODY
        # -----------------------------

        weight_match = re.search(r"Weight\s*:\s*(\d+\.?\d*)", text)
        weight = clean_number(weight_match.group(1)) if weight_match else None

        height_match = re.search(r"Height:\s*(\d+)\s*Feet\s*(\d+)", text)
        height_feet = clean_number(height_match.group(1)) if height_match else None
        height_inches = clean_number(height_match.group(2)) if height_match else None

        # STRICT FAIL CONDITION
        core_all_null = (
            k30_15 is None and
            valsalva is None and
            systolic is None and
            diastolic is None and
            dee is None
        )

        if core_all_null:
            return jsonify({"error": "report_format_not_supported"}), 422

        result = {
            "patient": {
                "first_name": first_name,
                "last_name": last_name,
                "dob": dob,
                "gender": gender,
                "exam_date": exam_date
            },
            "vitals": {
                "systolic_bp": systolic,
                "diastolic_bp": diastolic
            },
            "hrv": {
                "k30_15_ratio": k30_15,
                "valsava_ratio": valsalva
            },
            "metabolic": {
                "daily_energy_expenditure_kcal": dee
            },
            "body": {
                "weight_lbs": weight,
                "height_feet": height_feet,
                "height_inches": height_inches
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "v1_extract_report_service_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
