from flask import Flask, request, jsonify
import numpy as np
from pdf2image import convert_from_bytes
import cv2

app = Flask(__name__)

# =========================
# DECLARE (HARD CONSTANTS)
# =========================

ENGINE_NAME = "v62_vertical_band_height_filtered"
API_KEY = "ithrive_secure_2026_key"

DPI_LOCK = 200
EXPECTED_PROBE_HEIGHT = 2200
MAX_HEIGHT_DRIFT_RATIO = 0.03

VALUE_NONWHITE_THRESHOLD = 245

BAR_MIN_WIDTH = 600       # real disease bars are wide
BAR_MIN_HEIGHT = 14       # eliminate thin dividers
VERTICAL_SCAN_STEP = 2

TARGET_ROW_INDEX = 0      # <-- CHANGE THIS


# =========================
# APPLY
# =========================

def measure_vertical_band(value_channel, start_y, x_start, x_end, height):

    y_top = start_y
    y_bottom = start_y

    # scan upward
    while y_top > 0:
        row = value_channel[y_top, x_start:x_end]
        if np.mean(row) < VALUE_NONWHITE_THRESHOLD:
            y_top -= 1
        else:
            break

    # scan downward
    while y_bottom < height - 1:
        row = value_channel[y_bottom, x_start:x_end]
        if np.mean(row) < VALUE_NONWHITE_THRESHOLD:
            y_bottom += 1
        else:
            break

    return y_top, y_bottom


def detect_all_bars(img):

    runtime_height, runtime_width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    detected = []
    y = 10

    while y < runtime_height - 10:

        row = value[y, :]
        col_mask = row < VALUE_NONWHITE_THRESHOLD
        nonzero = np.where(col_mask)[0]

        if len(nonzero) > 0:
            x_start = nonzero[0]
            x_end = nonzero[-1]

            if (x_end - x_start) > BAR_MIN_WIDTH:

                y_top, y_bottom = measure_vertical_band(
                    value, y, x_start, x_end, runtime_height
                )

                band_height = y_bottom - y_top

                if band_height >= BAR_MIN_HEIGHT:
                    detected.append((y_top + y_bottom) // 2)
                    y = y_bottom + 10
                    continue

        y += VERTICAL_SCAN_STEP

    detected_sorted = sorted(detected)
    return detected_sorted


def compute_bar_fill(img, y_center):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    row = value[y_center, :]
    col_mask = row < VALUE_NONWHITE_THRESHOLD
    nonzero = np.where(col_mask)[0]

    if len(nonzero) == 0:
        return 0

    x_start = nonzero[0]
    x_end = nonzero[-1]

    y_top, y_bottom = measure_vertical_band(
        value, y_center, x_start, x_end, value.shape[0]
    )

    bar_region = value[y_top:y_bottom, x_start:x_end]

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
        return 0, 0

    if TARGET_ROW_INDEX >= len(bars):
        return len(bars), 0

    target_y = bars[TARGET_ROW_INDEX]
    percent = compute_bar_fill(img, target_y)

    return len(bars), percent


# =========================
# OUTPUT
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

        bars_detected, progression_percent = detect_progression_percent(img)

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
            "bars_detected": bars_detected,
            "results": results
        })

    except Exception as e:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
