import io
import re
import pdfplumber
from flask import Flask, request, jsonify

app = Flask(__name__)

AUTH_TOKEN = "ithrive_secure_2026_key"
BODY_PAGE_INDEX = 6


def safe_float(val):
    try:
        return float(val)
    except:
        return None


# ---------------------------------------------------
# AUTO-DETECT NUMERIC COLUMN
# ---------------------------------------------------
def extract_body_vector(pdf_bytes):
    body = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page = pdf.pages[BODY_PAGE_INDEX]
        words = page.extract_words()

        # Get only numeric tokens
        numeric_words = []
        for w in words:
            if re.fullmatch(r"\d+\.\d+|\d+", w["text"]):
                numeric_words.append({
                    "value": safe_float(w["text"]),
                    "x": w["x0"],
                    "y": w["top"]
                })

        if not numeric_words:
            return body

        # Group by approximate X (column clustering)
        numeric_words.sort(key=lambda w: w["x"])

        columns = []
        for word in numeric_words:
            placed = False
            for col in columns:
                if abs(word["x"] - col[0]["x"]) < 20:  # 20px tolerance
                    col.append(word)
                    placed = True
                    break
            if not placed:
                columns.append([word])

        # Select rightmost column
        right_column = max(columns, key=lambda col: col[0]["x"])

        # Sort top to bottom
        right_column.sort(key=lambda w: w["y"])

        values = [w["value"] for w in right_column if w["value"] is not None]

        # Expect order:
        # Total Body Water
        # Fat Free Mass
        # Weight
        if len(values) >= 3:
            body["total_body_water_lb"] = values[0]
            body["fat_free_mass_lb"] = values[1]
            body["weight_lb"] = values[2]

    return body


# ---------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------
def extract_report(pdf_bytes):
    result = {
        "body_composition": {},
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    result["body_composition"] = extract_body_vector(pdf_bytes)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            k_match = re.search(r"K30\/15.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
            if k_match:
                result["hrv"]["k30_15_ratio"] = safe_float(k_match.group(1))

            v_match = re.search(r"Valsalva.*?Value:\s*([0-9]+\.?[0-9]*)", text, re.DOTALL)
            if v_match:
                result["hrv"]["valsava_ratio"] = safe_float(v_match.group(1))

            dee_match = re.search(r"Daily Energy Expenditure[^0-9]*([0-9]+)", text)
            if dee_match:
                result["metabolic"]["daily_energy_expenditure_kcal"] = safe_float(dee_match.group(1))

            bp_match = re.search(
                r"Systolic\s*\/\s*Diastolic\s*pressure:\s*([0-9]+)\s*\/\s*([0-9]+)",
                text
            )
            if bp_match:
                result["vitals"]["systolic_bp"] = safe_float(bp_match.group(1))
                result["vitals"]["diastolic_bp"] = safe_float(bp_match.group(2))

    return result


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "COLUMN_AUTO_DETECT_ACTIVE"})


@app.route("/v1/extract-report", methods=["POST"])
def extract_report_endpoint():
    auth_header = request.headers.get("Authorization")

    if auth_header != f"Bearer {AUTH_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    try:
        result = extract_report(pdf_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
