import os
import re
import tempfile
import traceback
from collections import defaultdict
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
    return jsonify({"status": "FINAL_SIMPLE_DETERMINISTIC_PRODUCTION_ACTIVE"}), 200


# =========================================================
# UTILITIES
# =========================================================

BODY_LABELS = [
    "Weight",
    "Fat Free Mass",
    "Body Fat Mass",
    "Dry Lean Mass",
    "Total Body Water"
]


def is_number(text):
    return re.fullmatch(r"\d+(\.\d+)?", text) is not None


def group_words_by_row(words, tolerance=3):
    rows = defaultdict(list)
    for w in words:
        y = round(w["top"] / tolerance) * tolerance
        rows[y].append(w)
    return list(rows.values())


def extract_row_value(row_words):
    """
    Extract the correct lb value from a row:
    - collect all numeric values
    - keep realistic lb values (20–400)
    - return the largest realistic value
    """
    numbers = []

    for w in row_words:
        if is_number(w["text"]):
            val = float(w["text"])
            if 20 <= val <= 400:
                numbers.append(val)

    if not numbers:
        return None

    return max(numbers)


def extract_bp(full_text):
    systolic = None
    diastolic = None

    for line in full_text.split("\n"):
        lower = line.lower()
        if any(k in lower for k in ["blood", "pressure", "bp", "systolic"]):
            match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", line)
            if match:
                systolic = float(match.group(1))
                diastolic = float(match.group(2))

    return systolic, diastolic


def extract_hrv(full_text):
    result = {}

    for line in full_text.split("\n"):
        if "30/15" in line and ":" in line:
            nums = re.findall(r"\d+(\.\d+)?", line)
            if nums:
                result["k30_15_ratio"] = float(nums[-1])

        if "Valsalva" in line and ":" in line:
            nums = re.findall(r"\d+(\.\d+)?", line)
            if nums:
                result["valsava_ratio"] = float(nums[-1])

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

    full_text = ""

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:

            page_text = page.extract_text() or ""
            full_text += page_text

            if not any(label in page_text for label in BODY_LABELS):
                continue

            words = page.extract_words()
            rows = group_words_by_row(words)

            for row in rows:

                row_text = " ".join(w["text"] for w in row)

                if not any(label in row_text for label in BODY_LABELS):
                    continue

                value = extract_row_value(row)

                if value is None:
                    continue

                if "Weight" in row_text and "Target" not in row_text:
                    result["body_composition"]["weight_lb"] = value

                elif "Fat Free Mass" in row_text:
                    result["body_composition"]["fat_free_mass_lb"] = value

                elif "Body Fat Mass" in row_text:
                    result["body_composition"]["body_fat_mass_lb"] = value

                elif "Dry Lean Mass" in row_text:
                    result["body_composition"]["dry_lean_mass_lb"] = value

                elif "Total Body Water" in row_text:
                    result["body_composition"]["total_body_water_lb"] = value

                elif "Daily Energy Expenditure" in row_text:
                    result["metabolic"]["daily_energy_expenditure_kcal"] = value

    # Extract BP + HRV separately
    systolic, diastolic = extract_bp(full_text)
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
