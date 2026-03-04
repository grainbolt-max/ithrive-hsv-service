from flask import Flask, request, jsonify, send_file  
from pdf2image import convert_from_bytes  
import numpy as np  
import cv2  
import os  

app = Flask(name)

API_KEY = "ithrive_secure_2026_key"

# ======================================
# REFERENCE PAGE SIZE (FROM METADATA)
# ======================================

REF_WIDTH = 1700  
REF_HEIGHT = 2200  

# Stripe location in reference layout
REF_X_LEFT = 950  
REF_X_RIGHT = 1250  

# Disease row coordinates in reference layout
REF_DISEASE_ROWS = [
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

# ======================================
# AUTH
# ======================================

def require_auth(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split("Bearer ")[1].strip()
    return token == API_KEY


# ======================================
# SCALE COORDINATES
# ======================================

def scale_coordinates(page_width, page_height):

    scale_x = page_width / REF_WIDTH
    scale_y = page_height / REF_HEIGHT

    x_left = int(REF_X_LEFT * scale_x)
    x_right = int(REF_X_RIGHT * scale_x)

    rows = []

    for y1, y2 in REF_DISEASE_ROWS:
        rows.append((
            int(y1 * scale_y),
            int(y2 * scale_y)
        ))

    return x_left, x_right, rows


# ======================================
# DEBUG OVERLAY
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

    page_height, page_width = page.shape[:2]

    x_left, x_right, rows = scale_coordinates(page_width, page_height)

    for y1, y2 in rows:
        cv2.rectangle(page, (x_left, y1), (x_right, y2), (0, 0, 255), 3)

    output_path = "/tmp/debug_overlay.png"

    cv2.imwrite(output_path, page)

    return send_file(output_path, mimetype="image/png")


# ======================================
# HEALTH
# ======================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ITHRIVE HSV Service Running"})


if name == "main":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
