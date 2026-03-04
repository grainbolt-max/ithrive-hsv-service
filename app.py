from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import hashlib
import base64
import io

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

# Known canonical fingerprint
CANONICAL_HASH = "YlLY455AeeVlZXU8xGy1yd04QIomu+5OyCOaFw+8oHg="


def require_auth(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split("Bearer ")[1].strip()
    return token == API_KEY


# ----------------------------------------------------
# STABLE PANEL ANCHOR (GRAY HEADER DETECTION)
# ----------------------------------------------------
def find_panel_anchor(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 80])
    upper = np.array([180, 40, 140])

    mask = cv2.inRange(hsv, lower, upper)

    row_strength = np.sum(mask, axis=1)

    search_start = int(img.shape[0] * 0.15)

    anchor_y = search_start + np.argmax(row_strength[search_start:])

    return anchor_y


# ----------------------------------------------------
# LAYOUT OFFSETS (relative to anchor)
# ----------------------------------------------------
BASE_LAYOUT = {

    # PAGE 2
    "large_artery_stiffness": {"x": 1040, "y": 130, "w": 520, "h": 42},
    "peripheral_vessel": {"x": 1040, "y": 172, "w": 520, "h": 42},
    "blood_pressure_uncontrolled": {"x": 1040, "y": 214, "w": 520, "h": 42},
    "small_medium_artery": {"x": 1040, "y": 256, "w": 520, "h": 42},
    "atherosclerosis": {"x": 1040, "y": 298, "w": 520, "h": 42},
    "ldl_cholesterol": {"x": 1040, "y": 340, "w": 520, "h": 42},
    "lv_hypertrophy": {"x": 1040, "y": 382, "w": 520, "h": 42},

    "metabolic_syndrome": {"x": 1040, "y": 460, "w": 520, "h": 42},
    "insulin_resistance": {"x": 1040, "y": 502, "w": 520, "h": 42},
    "beta_cell_function": {"x": 1040, "y": 544, "w": 520, "h": 42},
    "blood_glucose": {"x": 1040, "y": 586, "w": 520, "h": 42},
    "tissue_inflammation": {"x": 1040, "y": 628, "w": 520, "h": 42},

    # PAGE 3
    "hypothyroidism": {"x": 1040, "y": -100, "w": 520, "h": 42},
    "hyperthyroidism": {"x": 1040, "y": -58, "w": 520, "h": 42},
    "hepatic_fibrosis": {"x": 1040, "y": -16, "w": 520, "h": 42},
    "chronic_hepatitis": {"x": 1040, "y": 26, "w": 520, "h": 42},

    "respiratory_disorders": {"x": 1040, "y": 106, "w": 520, "h": 42},
    "kidney_function": {"x": 1040, "y": 148, "w": 520, "h": 42},
    "digestive_disorders": {"x": 1040, "y": 190, "w": 520, "h": 42},

    "major_depression": {"x": 1040, "y": 300, "w": 520, "h": 42},
    "adhd_learning": {"x": 1040, "y": 342, "w": 520, "h": 42},
    "dopamine_decrease": {"x": 1040, "y": 384, "w": 520, "h": 42},
    "serotonin_decrease": {"x": 1040, "y": 426, "w": 520, "h": 42},
}


# ----------------------------------------------------
# PDF METADATA
# ----------------------------------------------------
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


# ----------------------------------------------------
# DEBUG OVERLAY
# ----------------------------------------------------
@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(images[1])

    anchor_y = find_panel_anchor(page)

    overlay = page.copy()

    for disease, coords in BASE_LAYOUT.items():

        x = coords["x"]
        y = anchor_y + coords["y"]
        w = coords["w"]
        h = coords["h"]

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


# ----------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ITHRIVE HSV Service Running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
