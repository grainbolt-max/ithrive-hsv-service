from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import io
import os

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

# --------------------------------------------------
# CURRENT TEST COORDINATES
# --------------------------------------------------

TEMPLATE_BAR_X = 900
TEMPLATE_BAR_Y = 540

BAR_WIDTH = 340
ROW_HEIGHT = 42
TOTAL_ROWS = 22


# --------------------------------------------------
# AUTH
# --------------------------------------------------

def require_auth(req):

    auth_header = req.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return False

    token = auth_header.split("Bearer ")[1].strip()

    return token == API_KEY


# --------------------------------------------------
# DRAW BIG COORDINATE GRID
# --------------------------------------------------

def draw_coordinate_grid(img):

    h, w = img.shape[:2]

    for x in range(0, w, 200):

        cv2.line(img, (x,0), (x,h), (180,180,180), 2)

        cv2.putText(
            img,
            f"X={x}",
            (x+5,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,0,255),
            3
        )

    for y in range(0, h, 200):

        cv2.line(img, (0,y), (w,y), (180,180,180), 2)

        cv2.putText(
            img,
            f"Y={y}",
            (20,y+40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,0,255),
            3
        )

    return img


# --------------------------------------------------
# DRAW CROSSHAIR AT RECTANGLE START
# --------------------------------------------------

def draw_crosshair(img,x,y):

    cv2.line(img,(x-20,y),(x+20,y),(0,255,0),3)
    cv2.line(img,(x,y-20),(x,y+20),(0,255,0),3)

    return img


# --------------------------------------------------
# DEBUG OVERLAY
# --------------------------------------------------

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

    overlay = draw_coordinate_grid(overlay)

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

        overlay = draw_crosshair(overlay,x,y)

        label = f"({x},{y})"

        cv2.putText(
            overlay,
            label,
            (x+10,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            3
        )

    _, buffer = cv2.imencode(".png",overlay)

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/png"
    )


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.route("/",methods=["GET"])
def health():
    return jsonify({"status":"ITHRIVE parser running"})


# --------------------------------------------------
# SERVER START
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
