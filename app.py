from flask import Flask, request, jsonify
import pdfplumber
import re
import os

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "ithrive_secure_2026_key")

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def require_auth(req):
    auth = req.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        return False
    return True


def extract_full_text(file_stream):
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += "\n" + page_text
    return text


def find_number(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


# ------------------------------------------------------------
# Deterministic Extractors
# ------------------------------------------------------------

def extract_patient(text):
    first_name = None
    last_name = None
    dob = None
    gender = None
    exam_date = None

    name_match = re.search(r"First/Last Name:\s*([A-Za-z]+)\s+([A-Za-z]+)", text)
    if name_match:
        first_name = name_match.group(1)
        last_name = name_match.group(2)

    dob_match = re.search(r"Date of birth:\s*([0-9\-\/]+)", text)
    if dob_match:
        dob = dob_match.group(1)

    gender_match = re.search(r"Gender:\s*(Male|Female)", text, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1)

    exam_match = re.search(r"Examination performed at:\s*([0-9\-\: ]+)", text)
    if exam_match:
        exam_date = exam_match.group(1).strip()

    return {
        "first_name": first_name,
        "last_name": last_name,
        "dob": dob,
        "gender": gender,
        "exam_date": exam_date
    }


def extract_body(text):
    weight_lbs = find_number(r"Weight\s*:\s*([0-9\.]+)", text)
    height_feet = find_number(r"Height:\s*([0-9]+)\s*Feet", text)
    height_inches = find_number(r"Feet\s*([0-9]+)\s*Inch", text)

    return {
        "weight_lbs": weight_lbs,
        "height_feet": height_feet,
        "height_inches": height_inches
    }


def extract_vitals(text):
    systolic = None
    diastolic = None

    bp_match = re.search(r"Systolic\s*/\s*Diastolic pressure:\s*([0-9]+)\s*/\s*([0-9]+)", text)
    if bp_match:
        systolic = float(bp_match.group(1))
        diastolic = float(bp_match.group(2))

    return {
        "systolic_bp": systolic,
        "diastolic_bp": diastolic
    }


def extract_hrv(text):
    k30 = find_number(r"K30\/15[\s\S]*?Value:\s*([0-9\.]+)", text)
    valsalva = find_number(r"Valsalva ratio[\s\S]*?Value:\s*([0-9\.]+)", text)

    return {
        "k30_15_ratio": k30,
        "valsava_ratio": valsalva
    }


def extract_metabolic(text):
    dee = find_number(r"Daily Energy Expenditure \(DEE\):\s*([0-9\.]+)", text)
    return {
        "daily_energy_expenditure_kcal": dee
    }


def extract_body_composition(text):
    return {
        "body_fat_percent": find_number(r"Body Fat\s*%[:\s]*([0-9\.]+)", text),
        "visceral_fat_rating": find_number(r"Visceral Fat Rating[:\s]*([0-9\.]+)", text),
        "muscle_mass_percent": find_number(r"Muscle Mass\s*%[:\s]*([0-9\.]+)", text),
        "skeletal_muscle_percent": find_number(r"Skeletal Muscle\s*%[:\s]*([0-9\.]+)", text),
        "total_body_water_percent": find_number(r"Total Body Water\s*%[:\s]*([0-9\.]+)", text),
        "intracellular_water_percent": find_number(r"Intra-Cellular Water\s*%[:\s]*([0-9\.]+)", text),
        "extracellular_water_percent": find_number(r"Extra-Cellular Water\s*%[:\s]*([0-9\.]+)", text),
        "metabolic_age": find_number(r"Metabolic Age[:\s]*([0-9\.]+)", text),
    }


DISEASE_FIELDS = [
    "atherosclerosis",
    "lv_hypertrophy",
    "large_artery_stiffness",
    "small_medium_artery_stiffness",
    "peripheral_vessels",
    "metabolic_syndrome",
    "insulin_resistance",
    "diabetes_screening",
    "ldl_cholesterol",
    "tissue_inflammatory_process",
    "hypothyroidism",
    "hyperthyroidism",
    "hepatic_fibrosis",
    "chronic_hepatitis",
    "respiratory",
    "kidney_function",
    "digestive_disorders",
    "major_depression"
]


def extract_disease(text):
    results = {}
    for field in DISEASE_FIELDS:
        readable = field.replace("_", " ")
        pattern = rf"{readable}[\s\S]*?([0-9]{{1,3}})\s*%"
        val = find_number(pattern, text)
        results[field] = val
    return results


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "extract_report_service_running"})


@app.route("/debug-text", methods=["POST"])
def debug_text():
    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    text = extract_full_text(file.stream)
    return jsonify({"text": text})


@app.route("/v1/extract-report", methods=["POST"])
def extract_report():
    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    text = extract_full_text(file.stream)

    patient = extract_patient(text)
    body = extract_body(text)
    vitals = extract_vitals(text)
    hrv = extract_hrv(text)
    metabolic = extract_metabolic(text)
    body_composition = extract_body_composition(text)
    disease = extract_disease(text)

    return jsonify({
        "patient": patient,
        "body": body,
        "vitals": vitals,
        "hrv": hrv,
        "metabolic": metabolic,
        "body_composition": body_composition,
        "disease": disease
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
