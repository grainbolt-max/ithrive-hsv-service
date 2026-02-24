from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes

app = Flask(__name__)

ENGINE_NAME = "v85_strict_stripe_engine"
API_KEY = "ithrive_secure_2026_key"

TARGET_PAGE_INDEX = 1
TOP_CROP_RATIO = 0.32
BOTTOM_CROP_RATIO = 0.06
ROW_COUNT = 24
LEFT_SCAN_RATIO = 0.65

# 🔒 STRICT thresholds
SAT_THRESHOLD = 150
VAL_THRESHOLD = 60
COLUMN_DENSITY_THRESHOLD = 0.20
MAX_GAP_COLUMNS = 2


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

    # ---- Crop out non-disease regions ----
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
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 🔒 STRICT colored mask
        colored_mask = (sat > SAT_THRESHOLD) & (val > VAL_THRESHOLD)

        stripe_end_col = 0
        gap_counter = 0
        stripe_active = False

        for col in range(scan_width):

            col_mask = colored_mask[:, col]
            col_density = np.sum(col_mask) / len(col_mask)

            if col_density > COLUMN_DENSITY_THRESHOLD:
                stripe_active = True
                stripe_end_col = col
                gap_counter = 0
            else:
                if stripe_active:
                    gap_counter += 1
                    if gap_counter > MAX_GAP_COLUMNS:
                        break

        stripe_width_ratio = stripe_end_col / scan_width if stripe_active else 0.0

        # Compute HSV only inside detected stripe
        if stripe_active and stripe_end_col > 0:
            stripe_pixels = colored_mask[:, :stripe_end_col]
            stripe_hsv = hsv[:, :stripe_end_col]
            valid_pixels = stripe_pixels > 0

            if np.sum(valid_pixels) > 0:
                avg_h = float(np.mean(stripe_hsv[:, :, 0][valid_pixels]))
                avg_s = float(np.mean(stripe_hsv[:, :, 1][valid_pixels]))
                avg_v = float(np.mean(stripe_hsv[:, :, 2][valid_pixels]))
            else:
                avg_h, avg_s, avg_v = 0.0, 0.0, 0.0
        else:
            avg_h, avg_s, avg_v = 0.0, 0.0, 0.0

        diagnostics.append({
            "row_index": i + 1,
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
