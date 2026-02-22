import io
import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

AUTH_TOKEN = "ithrive_secure_2026_key"
BODY_PAGE_INDEX = 6  # locked body composition page


def safe_float(val):
    try:
        return float(val)
    except:
        return None


# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
def preprocess_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    binary = gray.point(lambda x: 255 if x > 170 else 0, mode="1")
    return binary


# ---------------------------------------------------
# CROPPED OCR BODY EXTRACTION (RIGHT COLUMN ONLY)
# ---------------------------------------------------
def extract_body_cropped(pdf_bytes):
    body = {}

    try:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=300,
            first_page=BODY_PAGE_INDEX + 1,
            last_page=BODY_PAGE_INDEX + 1
        )

        if not images:
            return body

        image = images[0]
        width, height = image.size

        # Crop far-right value column only (removes reference ranges)
        cropped = image.crop((
            width * 0.70,   # left (shifted right)
            height * 0.08,  # top
            width * 0.95,   # right
            height * 0.45   # bottom
        ))

        processed = preprocess_image(cropped)

        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'

        text = pytesseract.image_to_string(processed, config=config)

        numbers = re.findall(r'\d+\.\d+|\d+', text)
        numbers = [safe_float(n) for n in numbers]
        numbers = [n for n in numbers if n is not None]

        # Expected order in right column (top to bottom):
        # Total Body Water
        # Fat Free Mass
        # Weight
        if len(numbers) >= 3:
            body["total_body_water_lb"] = numbers[0]
            body["fat_free_mass_lb"] = numbers[1]
            body["weight_lb"] = numbers[2]

    except Exception as e:
        print("CROPPED OCR ERROR:", e)

    return body


# ---------------------------------------------------
# MAIN REPORT EXTRACTION
# ---------------------------------------------------
def extract_report(pdf_bytes):
    result = {
        "body_composition": {},
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    result["body_composition"] = extract_body_cropped(pdf_bytes)

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
    return jsonify({"status": "CROPPED_RIGHT_COLUMN_LOCKED"})


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
