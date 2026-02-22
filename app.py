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
    return jsonify({"status": "FINAL_LOCKED_OCR_ACTIVE"}), 200


# =========================================================
# SAFE NUMERIC EXTRACTION
# =========================================================

def extract_numeric_after_label(text, label):
    """
    Finds first numeric value after a label.
    Hard-stable deterministic extraction.
    """
    pattern = rf"{label}[^0-9\-\.]*([-+]?\d*\.?\d+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


# =========================================================
# CORE EXTRACTION ENGINE
# =========================================================

def process_pdf(filepath):
    """
    Deterministic extraction.
    No printing.
    No streaming.
    No debug.
    Always returns structured dict.
    """

    result = {
        "body_composition": {},
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    with pdfplumber.open(filepath) as pdf:

        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += "\n" + text

    # -------------------------
    # Body Composition
    # -------------------------

    weight = extract_numeric_after_label(full_text, "Weight")
    fat_mass = extract_numeric_after_label(full_text, "Fat Mass")
    fat_free_mass = extract_numeric_after_label(full_text, "Fat Free Mass")
    tbw = extract_numeric_after_label(full_text, "Total Body Water")

    if weight:
        result["body_composition"]["weight_lb"] = weight
    if fat_mass:
        result["body_composition"]["fat_mass_lb"] = fat_mass
    if fat_free_mass:
        result["body_composition"]["fat_free_mass_lb"] = fat_free_mass
    if tbw:
        result["body_composition"]["total_body_water_lb"] = tbw

    # -------------------------
    # HRV
    # -------------------------

    k_ratio = extract_numeric_after_label(full_text, "30/15")
    valsalva = extract_numeric_after_label(full_text, "Valsalva")

    if k_ratio:
        result["hrv"]["k30_15_ratio"] = k_ratio
    if valsalva:
        result["hrv"]["valsava_ratio"] = valsalva

    # -------------------------
    # Metabolic
    # -------------------------

    daily_energy = extract_numeric_after_label(full_text, "Daily Energy Expenditure")

    if daily_energy:
        result["metabolic"]["daily_energy_expenditure_kcal"] = daily_energy

    # -------------------------
    # Vitals
    # -------------------------

    systolic = extract_numeric_after_label(full_text, "Systolic")
    diastolic = extract_numeric_after_label(full_text, "Diastolic")

    if systolic:
        result["vitals"]["systolic_bp"] = systolic
    if diastolic:
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

        # -------------------------
        # RETURN JSON ONLY
        # -------------------------
        return jsonify(result), 200

    except Exception as e:
        # HARD FAIL SAFE
        traceback.print_exc()
        return jsonify({
            "error": "Internal processing error"
        }), 500


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
