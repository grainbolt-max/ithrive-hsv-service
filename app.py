from flask import Flask, request, jsonify
import fitz
import numpy as np
import cv2

app = Flask(__name__)

ENGINE_NAME = "v104_band_diagnostic_engine"
API_KEY = "ithrive_secure_2026_key"
PAGE_INDEX = 1


def extract_image_by_index(page, image_index):
    images = page.get_images(full=True)
    if image_index >= len(images):
        return None

    xref = images[image_index][0]
    base_image = page.parent.extract_image(xref)
    image_bytes = base_image["image"]

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return img


def detect_horizontal_bands(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # saturation mask
    sat_mask = hsv[:, :, 1] > 80

    row_sums = sat_mask.sum(axis=1)

    bands = []
    in_band = False
    start = 0

    for i, val in enumerate(row_sums):
        if val > 50 and not in_band:
            in_band = True
            start = i
        elif val <= 50 and in_band:
            in_band = False
            end = i
            if end - start > 3:
                bands.append({
                    "start_y": int(start),
                    "end_y": int(end),
                    "height": int(end - start)
                })

    return bands


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def inspect():

    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    if len(doc) <= PAGE_INDEX:
        return jsonify({"error": "Page 2 not found"}), 400

    page = doc[PAGE_INDEX]

    panel1 = extract_image_by_index(page, 1)
    panel2 = extract_image_by_index(page, 2)

    if panel1 is None or panel2 is None:
        return jsonify({"error": "Disease panels not found"}), 400

    bands1 = detect_horizontal_bands(panel1)
    bands2 = detect_horizontal_bands(panel2)

    return jsonify({
        "engine": ENGINE_NAME,
        "panel1_height": int(panel1.shape[0]),
        "panel2_height": int(panel2.shape[0]),
        "panel1_detected_band_count": len(bands1),
        "panel2_detected_band_count": len(bands2),
        "panel1_bands": bands1,
        "panel2_bands": bands2
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
