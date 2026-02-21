import os
import re
import fitz
import pytesseract
import tempfile
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"

app = Flask(__name__)


# =========================
# AUTH
# =========================
def is_authorized(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return auth.split(" ")[1] == API_KEY


# =========================
# CLEAN NUMBER
# =========================
def clean_number(val):
    if not val:
        return None
    val = val.replace(",", ".")
    try:
        return float(val)
    except:
        return None


# =========================
# HRV PARSER
# =========================
def extract_hrv_from_text(text):
    text = text.replace(",", ".")
    text_lower = text.lower()

    rmssd_match = re.search(r"r\s*m\s*s\s*s\s*d.*?(\d+\.\d+|\d+)", text, re.IGNORECASE)
    lf_hf_match = re.search(r"l\s*f\s*[/\s]\s*h\s*f.*?(\d+\.\d+|\d+)", text, re.IGNORECASE)
    total_power_match = re.search(r"total\s*power.*?(\d+\.\d+|\d+)", text, re.IGNORECASE)

    rmssd = clean_number(rmssd_match.group(1)) if rmssd_match else None
    lf_hf = clean_number(lf_hf_match.group(1)) if lf_hf_match else None
    total_power = clean_number(total_power_match.group(1)) if total_power_match else None

    if rmssd is None or lf_hf is None:
        return None

    return {
        "rmssd_ms": rmssd,
        "lf_hf_ratio": lf_hf,
        "total_power_ms2": total_power
    }


# =========================
# LIGHT OCR FOR PAGE DETECTION
# =========================
def light_ocr(page):
    mat = fitz.Matrix(0.4, 0.4)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pix.save(tmp.name)
        path = tmp.name

    text = pytesseract.image_to_string(path, config="--psm 6 --oem 1")
    os.remove(path)
    return text


# =========================
# CROPPED HIGHER-QUALITY OCR (TOP REGION ONLY)
# =========================
def cropped_hrv_ocr(page):
    rect = page.rect

    # Top 45% of page
    crop_rect = fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x1,
        rect.y0 + rect.height * 0.45
    )

    mat = fitz.Matrix(0.9, 0.9)  # higher clarity but smaller region
    pix = page.get_pixmap(matrix=mat, clip=crop_rect, alpha=False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pix.save(tmp.name)
        path = tmp.name

    text = pytesseract.image_to_string(path, config="--psm 6 --oem 1")
    os.remove(path)
    return text


# =========================
# MAIN ENDPOINT
# =========================
@app.route("/extract-hrv", methods=["POST"])
def extract_hrv():
    if not is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "file_missing"}), 400

    try:
        file = request.files["file"]
        pdf_bytes = file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page in doc:
            # Step 1: Light detection OCR
            light_text = light_ocr(page).lower()

            if "heart rate variability" in light_text or "lf/hf" in light_text:
                # Step 2: High-quality cropped OCR
                detailed_text = cropped_hrv_ocr(page)
                hrv_data = extract_hrv_from_text(detailed_text)

                if hrv_data:
                    doc.close()
                    return jsonify(hrv_data)

        doc.close()
        return jsonify({"error": "hrv_not_detected"}), 422

    except Exception as e:
        return jsonify({
            "error": "processing_failed",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "region_based_hrv_extractor_running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
