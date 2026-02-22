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
    return jsonify({"status": "LINE_LOCKED_PRODUCTION_ACTIVE"}), 200


# =========================================================
# LINE LOCKED EXTRACTION HELPERS
# =========================================================

def extract_value_from_line(line, label):
    """
    Extract first numeric value from a line
    only if the label exists in that same line.
    """
    if label.lower() in line.lower():
        numbers = re.findall(r"[-+]?\d*\.?\d+", line)
        if numbers:
            try:
                return float(numbers[0])
            except:
                return None
    return None


def extract_bp_from_line(line):
    """
    Extract systolic/diastolic from a single line.
    Example: 110 / 73
    """
    if "bp" in line.lower() or "pressure" in line.lower():
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
                page_lines = text.split("\n")
                lines.extend(page_lines)

    # -----------------------------------------------------
    # Iterate line-by-line (LOCKED)
    # -----------------------------------------------------

    for line in lines:

        # -------------------------
        # Body Composition
        # -------------------------

        weight = extract_value_from_line(line, "Weight")
        if weight and "weight_lb" not in result["body_composition"]:
            result["body_composition"]["weight_lb"] = weight

        fat_mass = extract_value_from_line(line, "Fat Mass")
        if fat_mass:
            result["body_composition"]["fat_mass_lb"] = fat_mass

        fat_free = extract_value_from_line(line, "Fat Free Mass")
        if fat_free:
            result["body_composition"]["fat_free_mass_lb"] = fat_free

        tbw = extract_value_from_line(line, "Total Body Water")
        if tbw:
            result["body_composition"]["total_body_water_lb"] = tbw

        # -------------------------
        # HRV
        # -------------------------

        k_ratio = extract_value_from_line(line, "30/15")
        if k_ratio:
            result["hrv"]["k30_15_ratio"] = k_ratio

        valsalva = extract_value_from_line(line, "Valsalva")
        if valsalva:
            result["hrv"]["valsava_ratio"] = valsalva

        # -------------------------
        # Metabolic
        # -------------------------

        daily_energy = extract_value_from_line(line, "Daily Energy Expenditure")
        if daily_energy:
            result["metabolic"]["daily_energy_expenditure_kcal"] = daily_energy

        # -------------------------
        # Blood Pressure (Special Handling)
        # -------------------------

        systolic, diastolic = extract_bp_from_line(line)
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
