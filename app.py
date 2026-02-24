from flask import Flask, request, jsonify
import numpy as np
from pdf2image import convert_from_bytes
import cv2

app = Flask(__name__)

# =========================
# DECLARE (HARD CONSTANTS)
# =========================

ENGINE_NAME = "v59_deterministic_scaled_window_autoX"
API_KEY = "ithrive_secure_2026_key"

DPI_LOCK = 200

EXPECTED_PROBE_HEIGHT = 2200
EXPECTED_ANCHOR_Y = 1022

MAX_HEIGHT_DRIFT_RATIO = 0.03

BAR_SAMPLE_THICKNESS = 4
VERTICAL_SCAN_RANGE = 60

SATURATION_THRESHOLD = 30
MIN_BAR_WIDTH = 200  # fail if detected bar too narrow


# =========================
# APPLY (PURE GEOMETRY)
# =========================

def compute_progression_for_y(img, y):

    runtime_height, runtime_width, _ = img.shape

    y1 = y - BAR_SAMPLE_THICKNESS // 2
    y2 = y + BAR_SAMPLE_THICKNESS // 2

    if y1 < 0 or y2 >= runtime_height:
        return 0

    row_slice = img[y1:y2, :]

    hsv = cv2.cvtColor(row_slice, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Horizontal projection
    col_mask = np.mean(saturation, axis=0) > SATURATION_THRESHOLD

    nonzero_indices = np.where(col_mask)[0]

    if len(nonzero_indices) == 0:
        return 0

    x_start = int(nonzero_indices[0])
    x_end = int(nonzero_indices[-1])

    if (x_end - x_start) < MIN_BAR_WIDTH:
        return 0

    bar_region = saturation[:, x_start:x_end]

    filled_pixels = np.sum(bar_region > SATURATION_THRESHOLD)
    total_pixels = bar_region.size

    return int((filled_pixels / total_pixels) * 100)


def detect_progression_percent(img):

    runtime_height, runtime_width, _ = img.shape

    # Height validation
    height_ratio = abs(runtime_height - EXPECTED_PROBE_HEIGHT) / EXPECTED_PROBE_HEIGHT
    if height_ratio > MAX_HEIGHT_DRIFT_RATIO:
        raise RuntimeError(
            f"Height drift too large. Expected {EXPECTED_PROBE_HEIGHT}, got {runtime_height}"
        )

    scale = runtime_height / EXPECTED_PROBE_HEIGHT
    scaled_anchor = int(EXPECTED_ANCHOR_Y * scale)

    best_percent = 0

    for offset in range(-VERTICAL_SCAN_RANGE, VERTICAL_SCAN_RANGE + 1):
        y = scaled_anchor + offset

        if y < 10 or y > runtime_height - 10:
            continue

        percent = compute_progression_for_y(img, y)

        if percent > best_percent:
            best_percent = percent

    return best_percent


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
            "adhd_children_learning": {
                "progression_percent": progression_percent,
                "risk_label": "mild" if progression_percent > 0 else "none",
                "source": ENGINE_NAME
            }
        }

        return jsonify({
            "engine": ENGINE_NAME,
            "pages_found": pages_found,
            "results": results
        })

    except Exception as e:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
