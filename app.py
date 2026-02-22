import os
import re
import tempfile
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pdfplumber

# =========================================================
# CONFIG
# =========================================================

API_KEY = "ithrive_secure_2026_key"
MAX_FILE_SIZE_MB = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "FINAL_TABLE_LOCKED_PRODUCTION_ACTIVE"}), 200


# =========================================================
# EXTRACTION HELPERS
# =========================================================

def extract_last_number(line):
    numbers = re.findall(r"\d*\.?\d+", line)
    if numbers:
        return float(numbers[-1])
    return None


def extract_ratio_after_colon(line, label):
    if label.lower() in line.lower() and ":" in line:
        after_colon = line.split(":")[-1]
        numbers = re.findall(r"\d*\.?\d+", after_colon)
        if numbers:
            return float(numbers[-1])
    return None


def extract_bp_from_line(line):
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", line)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


# =========================================================
# CORE EXTRACTION ENGINE
# =========================================================

def process_pdf(filepath):

    result = {
        "body_composition": {},
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    with pdfplumber.open(filepath) as pdf:
        lines = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split("\n"))

    for line in lines:

        clean = line.strip()

        # =================================================
        # BODY COMPOSITION (TABLE SAFE)
        # =================================================

        if "Weight" in clean and "Target" not in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["weight_lb"] = value

        if "Fat Free Mass" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["fat_free_mass_lb"] = value

        if "Dry Lean Mass" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["dry_lean_mass_lb"] = value

        if "Body Fat Mass" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["body_fat_mass_lb"] = value

        if "Total Body Water" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["total_body_water_lb"] = value

        # =================================================
        # HRV
        # =================================================

        k_ratio = extract_ratio_after_colon(clean, "30/15")
        if k_ratio:
            result["hrv"]["k30_15_ratio"] = k_ratio

        valsalva = extract_ratio_after_colon(clean, "Valsalva")
        if valsalva:
            result["hrv"]["valsava_ratio"] = valsalva

        # =================================================
        # METABOLIC
        # =================================================

        if "Daily Energy Expenditure" in clean and ":" in clean:
            value = extract_last_number(clean)
            if value:
                result["metabolic"]["daily_energy_expenditure_kcal"] = value

        # =================================================
        # VITALS (BP)
        # =================================================

        systolic, diastolic = extract_bp_from_line(clean)
        if systolic and diastolic:
            result["vitals"]["systolic_bp"] = systolic
            result["vitals"]["diastolic_bp"] = diastolic

    return result


# =========================================================
# EXTRACTION ENDPOINT
# =========================================================

@app.route("/v1/extract-report", methods=["POST"])
def extract_report():

    try:

        # -------------------------
        # AUTH
        # -------------------------
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        # -------------------------
        # FILE VALIDATION
        # -------------------------
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)

        # -------------------------
        # TEMP SAVE
        # -------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # -------------------------
        # PROCESS
        # -------------------------
        result = process_pdf(tmp_path)

        # -------------------------
        # CLEANUP
        # -------------------------
        os.remove(tmp_path)

        return jsonify(result), 200

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal processing error"}), 500


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
