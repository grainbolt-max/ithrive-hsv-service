from flask import Flask, request, jsonify
import numpy as np
from pdf2image import convert_from_bytes
import cv2

app = Flask(__name__)

# =========================
# DECLARE (HARD CONSTANTS)
# =========================

ENGINE_NAME = "v61_full_vertical_scan_index_lock"
API_KEY = "ithrive_secure_2026_key"

DPI_LOCK = 200

EXPECTED_PROBE_HEIGHT = 2200
MAX_HEIGHT_DRIFT_RATIO = 0.03

VALUE_NONWHITE_THRESHOLD = 245
BAR_MIN_WIDTH = 300
BAR_SAMPLE_THICKNESS = 6
VERTICAL_SCAN_STEP = 3

TARGET_ROW_INDEX = 0  # <-- CHANGE THIS TO SELECT WHICH BAR (0 = top bar)


# =========================
# APPLY (FULL PAGE BAR DETECTION)
# =========================

def detect_all_bars(img):

    runtime_height, runtime_width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    detected_rows = []

    y = 10
    while y < runtime_height - 10:

        y1 = y - BAR_SAMPLE_THICKNESS // 2
        y2 = y + BAR_SAMPLE_THICKNESS // 2

        if y1 < 0 or y2 >= runtime_height:
            y += VERTICAL_SCAN_STEP
            continue

        row_slice = value[y1:y2, :]

        col_mask = np.mean(row_slice, axis=0) < VALUE_NONWHITE_THRESHOLD
        nonzero = np.where(col_mask)[0]

        if len(nonzero) > 0:
            x_start = nonzero[0]
            x_end = nonzero[-1]

            if (x_end - x_start) > BAR_MIN_WIDTH:
                detected_rows.append(y)
                y += 40  # skip downward to avoid duplicates
                continue

        y += VERTICAL_SCAN_STEP

    return sorted(detected_rows)


def compute_bar_fill(img, y):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    y1 = y - BAR_SAMPLE_THICKNESS // 2
    y2 = y + BAR_SAMPLE_THICKNESS // 2

    row_slice = value[y1:y2, :]

    col_mask = np.mean(row_slice, axis=0) < VALUE_NONWHITE_THRESHOLD
    nonzero = np.where(col_mask)[0]

    if len(nonzero) == 0:
        return 0

    x_start = nonzero[0]
    x_end = nonzero[-1]

    bar_region = value[y1:y2, x_start:x_end]

    filled_pixels = np.sum(bar_region < VALUE_NONWHITE_THRESHOLD)
    total_pixels = bar_region.size

    return int((filled_pixels / total_pixels) * 100)


def detect_progression_percent(img):

    runtime_height, runtime_width, _ = img.shape

    height_ratio = abs(runtime_height - EXPECTED_PROBE_HEIGHT) / EXPECTED_PROBE_HEIGHT
    if height_ratio > MAX_HEIGHT_DRIFT_RATIO:
        raise RuntimeError(
            f"Height drift too large. Expected {EXPECTED_PROBE_HEIGHT}, got {runtime_height}"
        )

    bars = detect_all_bars(img)

    if len(bars) == 0:
        return 0

    if TARGET_ROW_INDEX >= len(bars):
        return 0

    target_y = bars[TARGET_ROW_INDEX]

    return compute_bar_fill(img, target_y)


# =========================
# OUTPUT (API ROUTE)
# =========================

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

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    try:
        pages = convert_from_bytes(
            pdf_bytes,
            dpi=DPI_LOCK,
            fmt="png",
            single_file=False
        )

        pages_found = len(pages)
        img = np.array(pages[0])

        progression_percent = detect_progression_percent(img)

        results = {
            "selected_row": {
                "progression_percent": progression_percent,
                "risk_label": "mild" if progression_percent > 0 else "none",
                "source": ENGINE_NAME
            }
        }

        return jsonify({
            "engine": ENGINE_NAME,
            "pages_found": pages_found,
            "bars_detected": len(detect_all_bars(img)),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
