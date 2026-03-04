from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import io
import os

# ============================================================
# APP INIT
# ============================================================

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

# ============================================================
# TEMPLATE CONFIGURATION
# ============================================================

# Template image used to anchor alignment
TEMPLATE_FILE = "page2_template.png"

# Coordinates measured from the template image itself
# These correspond to the first disease score bar row
TEMPLATE_BAR_X = 980
TEMPLATE_BAR_Y = 420

BAR_WIDTH = 320
ROW_HEIGHT = 42
TOTAL_ROWS = 22

# ============================================================
# AUTH
# ============================================================

def require_auth(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth.split("Bearer ")[1].strip()
    return token == API_KEY


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def find_template_offset(page_img):

    template = cv2.imread(TEMPLATE_FILE)

    page_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(page_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    _, _, _, max_loc = cv2.minMaxLoc(result)

    offset_x = max_loc[0]
    offset_y = max_loc[1]

    return offset_x, offset_y


# ============================================================
# DEBUG OVERLAY ENDPOINT
# ============================================================

@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not require_auth(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    pdf_bytes = request.files["file"].read()

    pages = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(pages[1])

    offset_x, offset_y = find_template_offset(page)

    overlay = page.copy()

    for i in range(TOTAL_ROWS):

        x = TEMPLATE_BAR_X + offset_x
        y = TEMPLATE_BAR_Y + offset_y + (i * ROW_HEIGHT)

        cv2.rectangle(
            overlay,
            (x, y),
            (x + BAR_WIDTH, y + ROW_HEIGHT),
            (255, 0, 0),
            2
        )

    _, buffer = cv2.imencode(".png", overlay)

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/png"
    )


# ============================================================
# SCORE EXTRACTION
# ============================================================

@app.route("/v1/extract", methods=["POST"])
def extract():

    if not require_auth(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    pdf_bytes = request.files["file"].read()

    pages = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(pages[1])

    offset_x, offset_y = find_template_offset(page)

    scores = []

    for i in range(TOTAL_ROWS):

        x = TEMPLATE_BAR_X + offset_x
        y = TEMPLATE_BAR_Y + offset_y + (i * ROW_HEIGHT)

        region = page[y:y+ROW_HEIGHT, x:x+BAR_WIDTH]

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        lower = np.array([70, 50, 50])
        upper = np.array([170, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        pixels = cv2.countNonZero(mask)

        score = int((pixels / (BAR_WIDTH * ROW_HEIGHT)) * 100)

        scores.append(score)

    return jsonify({
        "engine": "template_anchor_v2",
        "scores": scores
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ITHRIVE HSV Service Running"})


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
