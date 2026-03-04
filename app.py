from flask import Flask, request, jsonify, send_file
import os
import numpy as np
from pdf2image import convert_from_bytes
import cv2
import io

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

# -------------------------
# PAGE 2 BAR PARAMETERS
# -------------------------

# MOVE LEFT INTO SCORE BAR COLUMN
BAR_X = 660
BAR_WIDTH = 300

ROW_HEIGHT = 42

START_Y = 360

TOTAL_ROWS = 22

# -------------------------
# AUTH
# -------------------------

def check_auth(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"

# -------------------------
# ROOT
# -------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ITHRIVE HSV Service Running"})

# -------------------------
# DEBUG OVERLAY
# -------------------------

@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not check_auth(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    pdf = request.files["file"].read()

    images = convert_from_bytes(pdf, dpi=200)

    page = np.array(images[1])

    overlay = page.copy()

    for i in range(TOTAL_ROWS):

        y = START_Y + (i * ROW_HEIGHT)

        cv2.rectangle(
            overlay,
            (BAR_X, y),
            (BAR_X + BAR_WIDTH, y + ROW_HEIGHT),
            (255, 0, 0),
            2
        )

    _, buffer = cv2.imencode(".png", overlay)

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/png"
    )

# -------------------------
# SCORE EXTRACTION
# -------------------------

@app.route("/v1/extract", methods=["POST"])
def extract():

    if not check_auth(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    pdf = request.files["file"].read()

    images = convert_from_bytes(pdf, dpi=200)

    page = np.array(images[1])

    results = []

    for i in range(TOTAL_ROWS):

        y = START_Y + (i * ROW_HEIGHT)

        crop = page[
            y:y + ROW_HEIGHT,
            BAR_X:BAR_X + BAR_WIDTH
        ]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            (80, 50, 50),
            (140, 255, 255)
        )

        ratio = np.sum(mask > 0) / mask.size

        score = int(ratio * 100)

        results.append(score)

    return jsonify({
        "engine": "v57_runtime_anchor_probe",
        "scores": results
    })

# -------------------------
# START SERVER
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
