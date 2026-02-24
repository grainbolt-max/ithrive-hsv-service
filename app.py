import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "geometry_probe_v1"
API_KEY = "ithrive_secure_2026_key"


@app.route("/")
def home():
    return "HSV Preprocess Service Running geometry_probe_v1"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        return jsonify({"error": "PDF conversion failed", "details": str(e)}), 500

    page_count = 0
    page_shapes = []

    for page in pages:
        page_count += 1

        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        # PRINT TO RENDER LOGS
        print("PAGE SHAPE:", image.shape)

        page_shapes.append({
            "height": image.shape[0],
            "width": image.shape[1]
        })

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": page_count,
        "page_shapes": page_shapes
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
