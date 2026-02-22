import io
import re
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

app = Flask(__name__)

AUTH_TOKEN = "ithrive_secure_2026_key"


def extract_report(pdf_bytes):
    result = {
        "body_composition": {},
        "hrv": {},
        "metabolic": {},
        "vitals": {}
    }

    # Convert ONLY Page 6 to image
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    page6 = pages[6]

    width, height = page6.size

    # LOCKED CROP REGION (Right column where numeric values live)
    crop_box = (
        int(width * 0.55),   # left
        int(height * 0.20),  # top
        int(width * 0.95),   # right
        int(height * 0.75)   # bottom
    )

    cropped = page6.crop(crop_box)

    # OCR
    ocr_text = pytesseract.image_to_string(cropped)

    # Extract numeric tokens
    numbers = re.findall(r"[0-9]+\.?[0-9]*", ocr_text)

    # Deterministic mapping by order (top-to-bottom appearance)
    # Adjusted to known report structure
    if len(numbers) >= 4:
        result["body_composition"] = {
            "weight_lb": float(numbers[0]),
            "fat_free_mass_lb": float(numbers[1]),
            "fat_mass_lb": float(numbers[2]),
            "total_body_water_lb": float(numbers[3])
        }

    return result


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "FINAL_LOCKED_OCR_ACTIVE"})


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
