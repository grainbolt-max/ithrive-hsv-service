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
# BODY COMPOSITION DEBUG (CORRECT PAGE MATCH)
# ------------------------------------------------------------

def debug_body_composition(file_stream):
    with pdfplumber.open(file_stream) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and "Body composition and follow up" in text:

                print("\n==============================")
                print("BODY COMPOSITION PAGE FOUND")
                print("Page index:", page_index)
                print("==============================\n")

                tables = page.extract_tables()
                print("TABLE COUNT:", len(tables))

                for t_index, table in enumerate(tables):
                    print("\n--- TABLE", t_index, "---")
                    for row in table:
                        print(row)

                print("\n==============================\n")
                break

    return None

# ------------------------------------------------------------
# HRV
# ------------------------------------------------------------

def extract_hrv(text):
    k30 = re.search(r"K30/15[\s\S]*?Value:\s*([0-9\.]+)", text)
    valsalva = re.search(r"Valsalva ratio[\s\S]*?Value:\s*([0-9\.]+)", text)

    return {
        "k30_15_ratio": float(k30.group(1)) if k30 else None,
        "valsava_ratio": float(valsalva.group(1)) if valsalva else None
    }

# ------------------------------------------------------------
# VITALS
# ------------------------------------------------------------

def extract_vitals(text):
    match = re.search(
        r"Systolic\s*/\s*Diastolic pressure:\s*([0-9]+)\s*/\s*([0-9]+)",
        text
    )

    if match:
        return {
            "systolic_bp": float(match.group(1)),
            "diastolic_bp": float(match.group(2))
        }

    return {
        "systolic_bp": None,
        "diastolic_bp": None
    }

# ------------------------------------------------------------
# METABOLIC
# ------------------------------------------------------------

def extract_metabolic(text):
    match = re.search(
        r"Daily Energy Expenditure \(DEE\):\s*([0-9\.]+)",
        text
    )

    if match:
        return {
            "daily_energy_expenditure_kcal": float(match.group(1))
        }

    return {
        "daily_energy_expenditure_kcal": None
    }

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

    # Extract text
    text = extract_full_text(file.stream)

    # Reset stream
    file.stream.seek(0)

    # DEBUG BODY PAGE
    debug_body_composition(file.stream)

    return jsonify({
        "vitals": extract_vitals(text),
        "hrv": extract_hrv(text),
        "metabolic": extract_metabolic(text),
        "body_composition": None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
