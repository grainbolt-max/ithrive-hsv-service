from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "v83_geometry_diagnostic_engine"
API_KEY = "ithrive_secure_2026_key"

TARGET_PAGE_INDEX = 1  # Page 2 visually (0-based index)

TOP_CROP_RATIO = 0.32      # remove Homeostasis section
BOTTOM_CROP_RATIO = 0.06   # remove legend color bar
ROW_COUNT = 24
LEFT_SCAN_RATIO = 0.65     # only scan left portion for stripe
SAT_THRESHOLD = 40         # minimum saturation to count as colored


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_bytes = request.files["file"].read()

    pages = convert_from_bytes(pdf_bytes, dpi=200)

    if TARGET_PAGE_INDEX >= len(pages):
        return jsonify({"error": "Target page index not found"}), 400

    page = pages[TARGET_PAGE_INDEX]
    page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    h, w, _ = page_img.shape

    # ---- Vertical crop ----
    top_crop = int(h * TOP_CROP_RATIO)
    bottom_crop = int(h * (1 - BOTTOM_CROP_RATIO))

    disease_region = page_img[top_crop:bottom_crop, :]

    region_h, region_w, _ = disease_region.shape
    row_height = region_h // ROW_COUNT

    diagnostics = []

    for i in range(ROW_COUNT):

        y1 = i * row_height
        y2 = (i + 1) * row_height if i < ROW_COUNT - 1 else region_h

        row_img = disease_region[y1:y2, :]

        scan_width = int(region_w * LEFT_SCAN_RATIO)
        scan_region = row_img[:, :scan_width]

        hsv = cv2.cvtColor(scan_region, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Mask colored pixels (exclude low saturation / white)
        colored_mask = (saturation > SAT_THRESHOLD) & (value > 40)

        colored_pixels = np.sum(colored_mask)
        total_pixels = colored_mask.size

        stripe_density = float(colored_pixels) / float(total_pixels)

        # Width ratio (how far stripe extends horizontally)
        col_sums = np.sum(colored_mask, axis=0)
        nonzero_cols = np.where(col_sums > 0)[0]

        if len(nonzero_cols) > 0:
            stripe_width_ratio = float(nonzero_cols.max()) / float(scan_width)
        else:
            stripe_width_ratio = 0.0

        # Average HSV of colored pixels
        if colored_pixels > 0:
            avg_h = float(np.mean(hsv[:, :, 0][colored_mask]))
            avg_s = float(np.mean(hsv[:, :, 1][colored_mask]))
            avg_v = float(np.mean(hsv[:, :, 2][colored_mask]))
        else:
            avg_h, avg_s, avg_v = 0.0, 0.0, 0.0

        diagnostics.append({
            "row_index": i + 1,
            "y_top_absolute": int(top_crop + y1),
            "y_bottom_absolute": int(top_crop + y2),
            "stripe_pixel_density": round(stripe_density, 4),
            "stripe_width_ratio": round(stripe_width_ratio, 4),
            "avg_hsv": [
                round(avg_h, 2),
                round(avg_s, 2),
                round(avg_v, 2)
            ]
        })

    return jsonify({
        "engine": ENGINE_NAME,
        "page_index_processed": TARGET_PAGE_INDEX,
        "rows_detected": ROW_COUNT,
        "diagnostics": diagnostics
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
