from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import hashlib
import base64
import io

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

CANONICAL_HASH = "YlLY455AeeVlZXU8xGy1yd04QIomu+5OyCOaFw+8oHg="


def require_auth(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split("Bearer ")[1].strip()
    return token == API_KEY


BASE_LAYOUT = {

    # PAGE 2 – CARDIO / DIABETES
    "large_artery_stiffness": {"x": 1040, "y": 750, "w": 520, "h": 42},
    "peripheral_vessel": {"x": 1040, "y": 792, "w": 520, "h": 42},
    "blood_pressure_uncontrolled": {"x": 1040, "y": 834, "w": 520, "h": 42},
    "small_medium_artery": {"x": 1040, "y": 876, "w": 520, "h": 42},
    "atherosclerosis": {"x": 1040, "y": 918, "w": 520, "h": 42},
    "ldl_cholesterol": {"x": 1040, "y": 960, "w": 520, "h": 42},
    "lv_hypertrophy": {"x": 1040, "y": 1002, "w": 520, "h": 42},

    "metabolic_syndrome": {"x": 1040, "y": 1080, "w": 520, "h": 42},
    "insulin_resistance": {"x": 1040, "y": 1122, "w": 520, "h": 42},
    "beta_cell_function": {"x": 1040, "y": 1164, "w": 520, "h": 42},
    "blood_glucose": {"x": 1040, "y": 1206, "w": 520, "h": 42},
    "tissue_inflammation": {"x": 1040, "y": 1248, "w": 520, "h": 42},

    # PAGE 3 – MISC DISEASES
    "hypothyroidism": {"x": 1040, "y": 520, "w": 520, "h": 42},
    "hyperthyroidism": {"x": 1040, "y": 562, "w": 520, "h": 42},
    "hepatic_fibrosis": {"x": 1040, "y": 604, "w": 520, "h": 42},
    "chronic_hepatitis": {"x": 1040, "y": 646, "w": 520, "h": 42},

    "respiratory_disorders": {"x": 1040, "y": 726, "w": 520, "h": 42},
    "kidney_function": {"x": 1040, "y": 768, "w": 520, "h": 42},
    "digestive_disorders": {"x": 1040, "y": 810, "w": 520, "h": 42},

    "major_depression": {"x": 1040, "y": 920, "w": 520, "h": 42},
    "adhd_learning": {"x": 1040, "y": 962, "w": 520, "h": 42},
    "dopamine_decrease": {"x": 1040, "y": 1004, "w": 520, "h": 42},
    "serotonin_decrease": {"x": 1040, "y": 1046, "w": 520, "h": 42},
}


@app.route("/v1/pdf-metadata", methods=["POST"])
def pdf_metadata():

    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    first_page = np.array(images[0])

    page_height, page_width = first_page.shape[:2]

    small = cv2.resize(first_page, (200, 200))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    sha = hashlib.sha256(gray.tobytes()).digest()

    pixel_hash_b64 = base64.b64encode(sha).decode("utf-8")

    return jsonify({
        "page_width": page_width,
        "page_height": page_height,
        "file_size": len(pdf_bytes),
        "page_count": len(images),
        "pixel_hash_b64": pixel_hash_b64
    })


@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(images[1])

    overlay = page.copy()

    for disease, c in BASE_LAYOUT.items():

        x = c["x"]
        y = c["y"]
        w = c["w"]
        h = c["h"]

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    _, buffer = cv2.imencode(".png", overlay)

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype="image/png"
    )


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ITHRIVE HSV Service Running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
