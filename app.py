from flask import Flask, request, jsonify
import pdfplumber
import re
import os

app = Flask(__name__)

API_KEY = os.environ.get("PREPROCESS_API_KEY", "ithrive_secure_2026_key")

# ------------------------------------------------------------
# AUTH
# ------------------------------------------------------------

def require_auth(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def to_float(val):
    try:
        return float(str(val).replace("%", "").strip())
    except:
        return None


def extract_percent(text):
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if match:
        return float(match.group(1))
    return None


# ------------------------------------------------------------
# TEXT EXTRACTION
# ------------------------------------------------------------

def extract_full_text(file_stream):
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += "\n" + page_text
    return text


# ------------------------------------------------------------
# STRICT BODY COMPOSITION PARSER (PAGE 7 FIXED FORMAT)
# ------------------------------------------------------------

def extract_body_composition(file_stream):
    result = {
        "intra_cellular_water_lb": None,
        "extra_cellular_water_lb": None,
        "dry_lean_mass_lb": None,
        "body_fat_mass_lb": None,
        "total_body_water_lb": None,
        "fat_free_mass_lb": None,
        "weight_lb": None,
        "fat_free_mass_percent": None,
        "body_fat_percent": None,
        "total_body_water_percent": None,
        "intra_cellular_water_percent": None,
        "extra_cellular_water_percent": None,
        "bmi": None,
        "basal_metabolic_rate_kcal": None
    }

    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            if "Body Composition Indicators (lb)" in text:

                tables = page.extract_tables()

                for table in tables:
                    for row in table:
                        if not row:
                            continue

                        row_label = str(row[0]).strip()

                        # ----- MAIN LB TABLE -----
                        if row_label == "Intra Cellular Water":
                            result["intra_cellular_water_lb"] = to_float(row[1])
                        if row_label == "Extra Cellular Water":
                            result["extra_cellular_water_lb"] = to_float(row[1])
                        if row_label == "Dry Lean Mass":
                            result["dry_lean_mass_lb"] = to_float(row[1])
                        if row_label == "Body Fat Mass":
                            result["body_fat_mass_lb"] = to_float(row[1])

                        # These are in wider row format
                        if row_label == "Total Body Water":
                            result["total_body_water_lb"] = to_float(row[2])
                        if row_label == "Fat Free Mass":
                            result["fat_free_mass_lb"] = to_float(row[3])
                        if row_label == "Weight":
                            result["weight_lb"] = to_float(row[4])

                        # ----- PERCENT ANALYSIS TABLE -----
                        if row_label == "Fat Free Mass" and "%" in str(row):
                            result["fat_free_mass_percent"] = extract_percent(str(row))
                        if row_label == "Body Fat Mass" and "%" in str(row):
                            result["body_fat_percent"] = extract_percent(str(row))
                        if row_label == "Total Body Water" and "%" in str(row):
                            result["total_body_water_percent"] = extract_percent(str(row))
                        if row_label == "Intra Cellular Water" and "%" in str(row):
                            result["intra_cellular_water_percent"] = extract_percent(str(row))
                        if row_label == "Extra Cellular Water" and "%" in str(row):
                            result["extra_cellular_water_percent"] = extract_percent(str(row))

                        # ----- BMI / BMR -----
                        if "Body Mass Index" in row_label:
                            result["bmi"] = to_float(row[1])

                        if "Basal Metabolic Rate" in row_label:
                            result["basal_metabolic_rate_kcal"] = to_float(row[1])

                break  # stop after page found

    return result


# ------------------------------------------------------------
# HRV + VITALS + METABOLIC
# ------------------------------------------------------------

def extract_hrv(text):
    k30 = re.search(r"K30/15[\s\S]*?Value:\s*([0-9\.]+)", text)
    valsalva = re.search(r"Valsalva ratio[\s\S]*?Value:\s*([0-9\.]+)", text)

    return {
        "k30_15_ratio": float(k30.group(1)) if k30 else None,
        "valsava_ratio": float(valsalva.group(1)) if valsalva else None
    }


def extract_vitals(text):
    match = re.search(r"Systolic\s*/\s*Diastolic pressure:\s*([0-9]+)\s*/\s*([0-9]+)", text)
    if match:
        return {
            "systolic_bp": float(match.group(1)),
            "diastolic_bp": float(match.group(2))
        }
    return {"systolic_bp": None, "diastolic_bp": None}


def extract_metabolic(text):
    match = re.search(r"Daily Energy Expenditure \(DEE\):\s*([0-9\.]+)", text)
    if match:
        return {"daily_energy_expenditure_kcal": float(match.group(1))}
    return {"daily_energy_expenditure_kcal": None}


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "extract_report_service_running"})


@app.route("/v1/extract-report", methods=["POST"])
def extract_report():
    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    text = extract_full_text(file.stream)

    file.stream.seek(0)
    body = extract_body_composition(file.stream)

    return jsonify({
        "vitals": extract_vitals(text),
        "hrv": extract_hrv(text),
        "metabolic": extract_metabolic(text),
        "body_composition": body
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
