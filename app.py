from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import hashlib
import base64
import json
import os

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"


def require_auth(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split("Bearer ")[1].strip()
    return token == API_KEY


# ===============================
# PDF METADATA ENDPOINT
# ===============================

@app.route("/v1/pdf-metadata", methods=["POST"])
def pdf_metadata():
    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        if not images:
            return jsonify({"error": "Unable to render PDF"}), 422

        first_page = np.array(images[0])
        page_height, page_width = first_page.shape[:2]
        file_size = len(pdf_bytes)
        page_count = len(images)

        small = cv2.resize(first_page, (200, 200))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        pixel_bytes = gray.tobytes()
        sha256_hash = hashlib.sha256(pixel_bytes).digest()
        pixel_hash_b64 = base64.b64encode(sha256_hash).decode("utf-8")

        return jsonify({
            "page_width": page_width,
            "page_height": page_height,
            "file_size": file_size,
            "page_count": page_count,
            "pixel_hash_b64": pixel_hash_b64
        })

    except Exception as e:
        return jsonify({
            "error": "PDF_METADATA_FAILURE",
            "message": str(e)
        }), 422


# ===============================
# HSV COLOR DETECTION
# ===============================

def detect_color_presence(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    lower = np.array([70, 50, 50])
    upper = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    color_pixels = cv2.countNonZero(mask)

    return color_pixels > 200


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():
    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    if "layout_profile" not in request.form:
        return jsonify({"error": "Missing layout_profile"}), 400

    layout_profile = json.loads(request.form["layout_profile"])

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        if len(images) < 2:
            return jsonify({"error": "PDF must contain at least 2 pages"}), 422

        page = np.array(images[1])

        panel_1 = {}
        panel_2 = {}
        color_hits = 0

        for disease_key, coords in layout_profile.items():
            x = coords["x"]
            y = coords["y"]
            w = coords["w"]
            h = coords["h"]

            region = page[y:y+h, x:x+w]

            if region.size == 0:
                continue

            has_color = detect_color_presence(region)

            if has_color:
                panel_1[disease_key] = "Moderate"
                color_hits += 1
            else:
                panel_1[disease_key] = "None/Low"

        REQUIRED_CONFIDENCE = 5

        if color_hits < REQUIRED_CONFIDENCE:
            return jsonify({
                "color_hits": color_hits,
                "error": "LAYOUT_MISMATCH",
                "reason": "Insufficient stripe confidence",
                "required": REQUIRED_CONFIDENCE
            }), 422

        return jsonify({
            "engine": "hsv_v7.1_strict",
            "panel_1": panel_1,
            "panel_2": panel_2
        })

    except Exception as e:
        return jsonify({
            "error": "HSV_PROCESSING_FAILURE",
            "message": str(e)
        }), 422


# ===============================
# DEBUG OVERLAY
# ===============================

@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():
    if not require_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    if "layout_profile" not in request.form:
        return jsonify({"error": "Missing layout_profile"}), 400

    layout_profile = json.loads(request.form["layout_profile"])

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        if len(images) < 2:
            return jsonify({"error": "PDF must contain at least 2 pages"}), 422

        page = np.array(images[1])

        for coords in layout_profile.values():
            x = coords["x"]
            y = coords["y"]
            w = coords["w"]
            h = coords["h"]
            cv2.rectangle(page, (x, y), (x + w, y + h), (0, 0, 255), 3)

        output_path = "/tmp/debug_overlay.png"
        cv2.imwrite(output_path, page)

        return send_file(output_path, mimetype="image/png")

    except Exception as e:
        return jsonify({
            "error": "DEBUG_OVERLAY_FAILURE",
            "message": str(e)
        }), 422


# ===============================
# TEST ROUTE
# ===============================

@app.route("/test-route", methods=["GET"])
def test_route():
    return jsonify({"message": "TEST ROUTE WORKING"})


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ITHRIVE HSV Service Running"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
