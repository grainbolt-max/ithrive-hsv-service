from flask import Flask, request, jsonify
import os
import numpy as np
from pdf2image import convert_from_bytes
import cv2

app = Flask(__name__)

ENGINE_NAME = "v57_runtime_anchor_probe"

API_KEY = "ithrive_secure_2026_key"


def find_first_nonwhite_row(img):
    """
    Find first row that is not pure white.
    """
    h, w, _ = img.shape
    for y in range(h):
        row = img[y, :, :]
        if np.mean(row) < 250:  # not white
            return y
    return None


def detect_gray_track_center(img):
    """
    Detect horizontal gray bar by scanning for low saturation band.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape

    for y in range(h):
        row = hsv[y, :, :]
        # Gray has low saturation
        if np.mean(row[:, 1]) < 20 and np.mean(row[:, 2]) < 240:
            return y
    return None


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    pdf_bytes = request.files["file"].read()

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        return jsonify({"error": "PDF conversion failed", "details": str(e)})

    img = np.array(pages[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width, _ = img.shape

    first_nonwhite = find_first_nonwhite_row(img)
    gray_center = detect_gray_track_center(img)

    if first_nonwhite is None:
        return jsonify({"error": "No nonwhite row found"})

    if gray_center is None:
        return jsonify({"error": "No gray band found"})

    # Estimate row height by scanning next gray band
    next_gray = None
    for y in range(gray_center + 10, height):
        row = img[y, :, :]
        if np.mean(row) < 240:
            next_gray = y
            break

    row_height = None
    if next_gray:
        row_height = next_gray - gray_center

    return jsonify({
        "engine": ENGINE_NAME,
        "image_height": height,
        "image_width": width,
        "first_nonwhite_y": first_nonwhite,
        "first_gray_track_y": gray_center,
        "estimated_row_height": row_height
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
