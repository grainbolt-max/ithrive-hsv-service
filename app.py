from flask import Flask, request, jsonify, send_file
import os
import numpy as np
from pdf2image import convert_from_bytes
import cv2
import hashlib
import base64
from parser.router import choose_parser

app = Flask(__name__)

ENGINE_NAME = "v57_runtime_anchor_probe"
API_KEY = "ithrive_secure_2026_key"

def require_auth(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split("Bearer ")[1].strip()
    return token == API_KEY

def find_first_nonwhite_row(img):
    h, w, _ = img.shape
    for y in range(h):
        row = img[y:y+1, :]
        if np.mean(row) < 250:
            return y
    return 0

def detect_bar_color(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower = np.array([70,50,50])
    upper = np.array([170,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    pixels = cv2.countNonZero(mask)
    return pixels > 200

@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():
    if not require_auth(request):
        return jsonify({"error":"unauthorized"}), 401
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    pdf_bytes = request.files["file"].read()
    images = convert_from_bytes(pdf_bytes)
    if len(images) < 2:
        return jsonify({"error":"page 2 missing"}), 400
    img = np.array(images[1])
    parser_choice = choose_parser(img)
    print("Detected parser:", parser_choice)
    h, w, _ = img.shape
    BAR_X = 760
    BAR_WIDTH = 340
    overlay = img.copy()
    for y in range(500, 2000, 40):
        region = img[y:y+40, BAR_X:BAR_X+BAR_WIDTH]
        if region.size == 0:
            continue
        detected = detect_bar_color(region)
        color = (0,255,0) if detected else (0,0,255)
        cv2.rectangle(
            overlay,
            (BAR_X,y),
            (BAR_X+BAR_WIDTH,y+40),
            color,
            2
        )
    output_path = "debug_overlay.png"
    cv2.imwrite(output_path, overlay)
    return send_file(output_path, mimetype="image/png")

@app.route("/v1/pdf-metadata", methods=["POST"])
def pdf_metadata():
    if not require_auth(request):
        return jsonify({"error":"unauthorized"}), 401
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    pdf_bytes = request.files["file"].read()
    images = convert_from_bytes(pdf_bytes)
    page = np.array(images[1])
    small = cv2.resize(page,(200,200))
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    pixel_hash = hashlib.sha256(gray.tobytes()).digest()
    pixel_hash_b64 = base64.b64encode(pixel_hash).decode()
    return jsonify({
        "engine": ENGINE_NAME,
        "pixel_hash_b64": pixel_hash_b64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
