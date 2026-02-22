import io
import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from flask import Flask, request, jsonify

app = Flask(__name__)

AUTH_TOKEN = "ithrive_secure_2026_key"


def safe_float(val):
    try:
        return float(val)
    except:
        return None


# ---------------------------------------------------
# OCR BODY PAGE (Auto-detect correct page)
# ---------------------------------------------------
def ocr_body_page(pdf_bytes):
    try:
        # First detect which page contains body composition
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            body_page_index = None

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and "Body Composition" in text:
                    body_page_index = i
                    break

        if body_page_index is None:
            return {}

        # OCR only that specific page
        images = convert_from_bytes(
            pdf_bytes,
            dpi=150,
            first_page=body_page_index + 1,
            last_page=body_page_index + 1
        )

        for image in images:
            text = pytesseract.image_to_string(image)

            body = {}

            patterns = {
                "weight_lb": r"Weight[^0-9]*([0-9]+\.?[0-9]*)",
                "fat_free_mass_lb": r"Fat\s*Free\s*Mass[^0-9]*([0-9]+\.?[0-9]*)",
                "fat_mass_lb": r"(Body\s*Fat\s*Mass|Fat\s*Mass)[^0-9]*([0-9]+\.?[0-9]*)",
                "total_body_water_lb": r"Total\s*Body\s*Water[^0-9]*([0-9]+\.?[0-9]*)",
                "body_fat_percent": r"(Percent\s*Body\s*Fat|Body\s*Fat)[^0-9]*([0-9]+\.?[0-9]*)"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    body[key] = safe_float(match.groups()[-1])

            return body

    except Exception as e:
        print("OCR ERROR:", e)

    return {}


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

    # OCR body
    result["body_composition"] = ocr_body_page(pdf_bytes)

    # Text extraction for other values
    try:
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

    except Exception as e:
        print("TEXT EXTRACTION ERROR:", e)

    return result


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "FINAL_DYNAMIC_OCR_VERSION_ACTIVE"})


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
        print("ENDPOINT ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
