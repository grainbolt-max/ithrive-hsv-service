from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import os

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

# ======================================
# HARDCODED STRIPE POSITION (FIXED)
# ======================================

# These are now aligned with the actual disease bars
X_LEFT = 950
X_RIGHT = 1250

DISEASE_ROWS = [
    (689, 709),
    (714, 734),
    (739, 759),
    (764, 784),
    (789, 809),
    (814, 834),
    (839, 859),
    (874, 894),
    (899, 919),
    (924, 944),
    (949, 969),
    (974, 994),
    (1145, 1165),
    (1170, 1190),
    (1195, 1215),
    (1215, 1235),
    (1235, 1255),
    (1260, 1280),
    (1285, 1305),
    (1310, 1330),
    (1355, 1375),
    (1380, 1400),
    (1405, 1425),
    (1425, 1445)
]


def require_auth(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split("Bearer ")[1].strip()
    return token == API_KEY


# ======================================
# DEBUG OVERLAY — HARD CODED
# ======================================

@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():
    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    if len(images) < 2:
        return jsonify({"error": "PDF must contain at least 2 pages"}), 422

    page = np.array(images[1])

    for (y1, y2) in DISEASE_ROWS:
        cv2.rectangle(page, (X_LEFT, y1), (X_RIGHT, y2), (0, 0, 255), 3)

    output_path = "/tmp/debug_overlay.png"
    cv2.imwrite(output_path, page)

    return send_file(output_path, mimetype="image/png")


# ======================================
# HEALTH CHECK
# ======================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "HSV SERVICE RUNNING"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
