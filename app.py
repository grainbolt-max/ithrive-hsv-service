from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import io
import os

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

# ---------------------------------------------------------
# CURRENT TEST RECTANGLE
# ---------------------------------------------------------

TEMPLATE_BAR_X = 900
TEMPLATE_BAR_Y = 540

BAR_WIDTH = 340
ROW_HEIGHT = 42
TOTAL_ROWS = 22


# ---------------------------------------------------------
# AUTH
# ---------------------------------------------------------

def require_auth(req):

    auth_header = req.headers.get("Authorization","")

    if not auth_header.startswith("Bearer "):
        return False

    token = auth_header.split("Bearer ")[1].strip()

    return token == API_KEY


# ---------------------------------------------------------
# DRAW GRID
# ---------------------------------------------------------

def draw_grid(img):

    h, w = img.shape[:2]

    for x in range(0, w, 25):

        color = (0,200,0)  # green minor grid
        thickness = 1

        if x % 100 == 0:
            color = (0,0,255)  # red major grid
            thickness = 2

        cv2.line(img,(x,0),(x,h),color,thickness)

        if x % 100 == 0:
            cv2.putText(
                img,
                str(x),
                (x+5,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

    for y in range(0, h, 25):

        color = (0,200,0)
        thickness = 1

        if y % 100 == 0:
            color = (0,0,255)
            thickness = 2

        cv2.line(img,(0,y),(w,y),color,thickness)

        if y % 100 == 0:
            cv2.putText(
                img,
                str(y),
                (5,y+30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

    return img


# ---------------------------------------------------------
# DEBUG OVERLAY
# ---------------------------------------------------------

@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not require_auth(request):
        return jsonify({"error":"unauthorized"}),401

    if "file" not in request.files:
        return jsonify({"error":"missing file"}),400

    pdf_bytes = request.files["file"].read()

    pages = convert_from_bytes(pdf_bytes,dpi=200)

    page = np.array(pages[1])

    overlay = page.copy()

    overlay = draw_grid(overlay)

    for i in range(TOTAL_ROWS):

        x = TEMPLATE_BAR_X
        y = TEMPLATE_BAR_Y + (i * ROW_HEIGHT)

        cv2.rectangle(
            overlay,
            (x,y),
            (x+BAR_WIDTH,y+ROW_HEIGHT),
            (255,0,0),
            3
        )

        cv2.putText(
            overlay,
            f"({x},{y})",
            (x+5,y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )

    _, buffer = cv2.imencode(".png",overlay)

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/png"
    )


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.route("/")
def health():
    return jsonify({"status":"ITHRIVE parser running"})


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
