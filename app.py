import os
import re
import tempfile
import traceback
from collections import defaultdict
from statistics import median
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
    return jsonify({"status": "GENERIC_COLUMN_LOCKED_PRODUCTION_ACTIVE"}), 200


# =========================================================
# UTILITIES
# =========================================================

def is_number(text):
    return re.fullmatch(r"\d+(\.\d+)?", text) is not None


def group_words_by_row(words, tolerance=3):
    rows = defaultdict(list)
    for w in words:
        y = round(w["top"] / tolerance) * tolerance
        rows[y].append(w)
    return rows.values()


def detect_numeric_column(words):
    numeric_x = []

    for w in words:
        if is_number(w["text"]):
            numeric_x.append(w["x0"])

    if not numeric_x:
        return None

    # cluster via median band detection
    center = median(numeric_x)

    # determine band (±40px safe window)
    return (center - 40, center + 40)


def detect_label_column(words):
    text_x = []

    for w in words:
        if not is_number(w["text"]):
            text_x.append(w["x0"])

    if not text_x:
        return None

    center = median(text_x)
    return (center - 80, center + 80)


def extract_bp_from_text(full_text):
    systolic = None
    diastolic = None

    lines = full_text.split("\n")

    for line in lines:
        lower = line.lower()
        if any(k in lower for k in ["blood", "pressure", "bp", "systolic"]):
            match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", line)
            if match:
                systolic = float(match.group(1))
                diastolic = float(match.group(2))

    return systolic, diastolic


def extract_hrv(full_text):
    result = {}

    lines = full_text.split("\n")

    for line in lines:
        if "30/15" in line and ":" in line:
            numbers = re.findall(r"\d+(\.\d+)?", line)
            if numbers:
                result["k30_15_ratio"] = float(numbers[-1])

        if "Valsalva" in line and ":" in line:
            numbers = re.findall(r"\d+(\.\d+)?", line)
            if numbers:
                result["valsava_ratio"] = float(numbers[-1])

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

    all_words = []
    full_text = ""

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() or ""
            all_words.extend(page.extract_words())

    if not all_words:
        return result

    value_band = detect_numeric_column(all_words)
    label_band = detect_label_column(all_words)

    if not value_band or not label_band:
        return result

    rows = group_words_by_row(all_words)

    for row in rows:

        label_parts = []
        value = None

        for w in row:
            x = w["x0"]
            text = w["text"]

            if label_band[0] <= x <= label_band[1] and not is_number(text):
                label_parts.append(text)

            if value_band[0] <= x <= value_band[1] and is_number(text):
                value = float(text)

        if not label_parts or value is None:
            continue

        label = " ".join(label_parts)

        # BODY COMPOSITION
        if "Weight" in label and "Target" not in label:
            result["body_composition"]["weight_lb"] = value

        elif "Fat Free Mass" in label:
            result["body_composition"]["fat_free_mass_lb"] = value

        elif "Body Fat Mass" in label:
            result["body_composition"]["body_fat_mass_lb"] = value

        elif "Dry Lean Mass" in label:
            result["body_composition"]["dry_lean_mass_lb"] = value

        elif "Total Body Water" in label:
            result["body_composition"]["total_body_water_lb"] = value

        elif "Daily Energy Expenditure" in label:
            result["metabolic"]["daily_energy_expenditure_kcal"] = value

    # Separate extraction (never table-based)
    systolic, diastolic = extract_bp_from_text(full_text)
    if systolic and diastolic:
        result["vitals"]["systolic_bp"] = systolic
        result["vitals"]["diastolic_bp"] = diastolic

    result["hrv"] = extract_hrv(full_text)

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
