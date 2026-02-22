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
# TABLE EXTRACTION (DETERMINISTIC BODY COMPOSITION)
# ------------------------------------------------------------

def extract_body_composition_tables(file_stream):
    results = {
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
            if text and "Body Composition Indicators (lb)" in text:
                tables = page.extract_tables()

                for table in tables:
                    for row in table:
                        if not row:
                            continue

                        row_text = " ".join([str(cell) for cell in row if cell])

                        # ----- LB VALUES -----
                        if "Intra Cellular Water" in row_text:
                            results["intra_cellular_water_lb"] = safe_float(row)
                        if "Extra Cellular Water" in row_text:
                            results["extra_cellular_water_lb"] = safe_float(row)
                        if "Dry Lean Mass" in row_text:
                            results["dry_lean_mass_lb"] = safe_float(row)
                        if "Body Fat Mass" in row_text:
                            results["body_fat_mass_lb"] = safe_float(row)
                        if "Total Body Water" in row_text:
                            results["total_body_water_lb"] = safe_float(row)
                        if "Fat Free Mass" in row_text:
                            results["fat_free_mass_lb"] = safe_float(row)
                        if "Weight" in row_text and results["weight_lb"] is None:
                            results["weight_lb"] = safe_float(row)

                        # ----- PERCENT VALUES -----
                        if "%" in row_text:
                            if "Fat Free Mass" in row_text:
                                results["fat_free_mass_percent"] = extract_percent(row_text)
                            if "Body Fat Mass" in row_text:
                                results["body_fat_percent"] = extract_percent(row_text)
                            if "Total Body Water" in row_text:
                                results["total_body_water_percent"] = extract_percent(row_text)
                            if "Intra Cellular Water" in row_text:
                                results["intra_cellular_water_percent"] = extract_percent(row_text)
                            if "Extra Cellular Water" in row_text:
                                results["extra_cellular_water_percent"] = extract_percent(row_text)

                        if "Body Mass Index" in row_text:
                            results["bmi"] = extract_number(row_text)

                        if "Basal Metabolic Rate" in row_text:
                            results["basal_metabolic_rate_kcal"] = extract_number(row_text)

                break  # stop once page found

    return results


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def safe_float(row):
    for cell in row:
        try:
            return float(cell)
        except:
            continue
    return None


def extract_percent(text):
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if match:
        return float(match.group(1))
    return None


def extract_number(text):
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    return None


# ------------------------------------------------------------
# HRV + VITALS + METABOLIC (UNCHANGED)
# ------------------------------------------------------------

def extract_hrv(text):
    k30 = extract_number_section(text, "K30/15")
    valsalva = extract_number_section(text, "Valsalva ratio")

    return {
        "k30_15_ratio": k30,
        "valsava_ratio": valsalva
    }


def extract_number_section(text, section_name):
    pattern = rf"{section_name}[\s\S]*?Value:\s*([0-9\.]+)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


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

    # reset stream for table parsing
    file.stream.seek(0)
    body_comp = extract_body_composition_tables(file.stream)

    hrv = extract_hrv(text)
    vitals = extract_vitals(text)
    metabolic = extract_metabolic(text)

    return jsonify({
        "vitals": vitals,
        "hrv": hrv,
        "metabolic": metabolic,
        "body_composition": body_comp
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
