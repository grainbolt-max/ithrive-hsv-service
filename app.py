from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import io
import os

# ============================================================
# CONFIG
# ============================================================

API_KEY = "ithrive_secure_2026_key"

TEMPLATE_PATH = "page2_template.png"

ROW_HEIGHT = 42
TOTAL_ROWS = 22
BAR_WIDTH = 340

# ============================================================
# APP
# ============================================================

app = Flask(__name__)

# ============================================================
# AUTH
# ============================================================

def require_auth(req):

    auth_header = req.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return False

    token = auth_header.split("Bearer ")[1].strip()

    return token == API_KEY


# ============================================================
# TEMPLATE ALIGNMENT
# ============================================================

def find_template_offset(page):

    template = cv2.imread(TEMPLATE_PATH)

    page_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(page_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    _, _, _, max_loc = cv2.minMaxLoc(result)

    offset_x = max_loc[0]
    offset_y = max_loc[1]

    return offset_x, offset_y


# ============================================================
# AUTO DETECT BAR COLUMN
# ============================================================

def detect_bar_column(template):

    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # detect the cyan/blue score bars
    lower = np.array([70, 40, 40])
    upper = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    projection = np.sum(mask, axis=0)

    bar_x = np.argmax(projection)

    return bar_x


# ============================================================
# AUTO DETECT FIRST ROW
# ============================================================

def detect_first_row(template):

    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    projection = np.sum(edges, axis=1)

    first_row = np.argmax(projection)

    return first_row


# ============================================================
# DEBUG OVERLAY
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

    template = cv2.imread(TEMPLATE_PATH)

    offset_x, offset_y = find_template_offset(page)

    bar_x_template = detect_bar_column(template)

    row_y_template = detect_first_row(template)

    overlay = page.copy()

    for i in range(TOTAL_ROWS):

        x = offset_x + bar_x_template
        y = offset_y + row_y_template + (i * ROW_HEIGHT)

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
# EXTRACT SCORES
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

    template = cv2.imread(TEMPLATE_PATH)

    offset_x, offset_y = find_template_offset(page)

    bar_x_template = detect_bar_column(template)

    row_y_template = detect_first_row(template)

    scores = []

    for i in range(TOTAL_ROWS):

        x = offset_x + bar_x_template
        y = offset_y + row_y_template + (i * ROW_HEIGHT)

        region = page[y:y+ROW_HEIGHT, x:x+BAR_WIDTH]

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        lower = np.array([70, 40, 40])
        upper = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        pixels = cv2.countNonZero(mask)

        score = int((pixels / (BAR_WIDTH * ROW_HEIGHT)) * 100)

        scores.append(score)

    return jsonify({
        "engine": "template_locked_parser_v1",
        "scores": scores
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/")
def health():

    return jsonify({
        "status": "ITHRIVE HSV parser running"
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
