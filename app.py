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
    return jsonify({"status": "RESTORED_SIMPLE_TEXT_EXTRACTION_ACTIVE"}), 200


# =========================================================
# UTILITIES
# =========================================================

def extract_last_number(line):
    numbers = re.findall(r"\d+\.?\d*", line)
    if numbers:
        return float(numbers[-1])
    return None


def extract_bp(lines):
    systolic = None
    diastolic = None

    for line in lines:
        lower = line.lower()
        if any(k in lower for k in ["blood", "pressure", "bp", "systolic"]):
            match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", line)
            if match:
                systolic = float(match.group(1))
                diastolic = float(match.group(2))

    return systolic, diastolic


def extract_hrv(lines):
    result = {}

    for line in lines:
        if "30/15" in line and ":" in line:
            result["k30_15_ratio"] = extract_last_number(line)

        if "Valsalva" in line and ":" in line:
            result["valsava_ratio"] = extract_last_number(line)

    return result


# =========================================================
# CORE ENGINE
# =========================================================

def process_pdf(filepath):

    result = {
        "body_composition": {},
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    lines = []

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split("\n"))

    for line in lines:

        clean = line.strip()

        # =========================
        # BODY COMPOSITION
        # =========================

        if "Weight" in clean and "Target" not in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["weight_lb"] = value

        elif "Fat Free Mass" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["fat_free_mass_lb"] = value

        elif "Body Fat Mass" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["body_fat_mass_lb"] = value

        elif "Dry Lean Mass" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["dry_lean_mass_lb"] = value

        elif "Total Body Water" in clean:
            value = extract_last_number(clean)
            if value:
                result["body_composition"]["total_body_water_lb"] = value

        # =========================
        # METABOLIC
        # =========================

        elif "Daily Energy Expenditure" in clean:
            value = extract_last_number(clean)
            if value:
                result["metabolic"]["daily_energy_expenditure_kcal"] = value

    # =========================
    # VITALS + HRV (SEPARATE)
    # =========================

    systolic, diastolic = extract_bp(lines)
    if systolic and diastolic:
        result["vitals"]["systolic_bp"] = systolic
        result["vitals"]["diastolic_bp"] = diastolic

    result["hrv"] = extract_hrv(lines)

    return result


# =========================================================
# ENDPOINT
# =========================================================

@app.route("/v1/extract-report", methods=["POST"])
def extract_report():
    try:
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        secure_name = secure_filename(file.filename)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result = process_pdf(tmp_path)

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
