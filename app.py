from flask import Flask, request, jsonify
import numpy as np
from pdf2image import convert_from_bytes
import cv2

app = Flask(__name__)

# =========================
# DECLARE (HARD CONSTANTS)
# =========================

ENGINE_NAME = "v57_deterministic_scaled_anchor"
API_KEY = "ithrive_secure_2026_key"

DPI_LOCK = 200

EXPECTED_PROBE_WIDTH = 1700
EXPECTED_PROBE_HEIGHT = 2200

EXPECTED_ANCHOR_Y = 1022  # Derived from geometry probe at 2200px height
MAX_HEIGHT_DRIFT_RATIO = 0.03  # 3% hard fail

BAR_X_START = 400
BAR_X_END = 1500
BAR_SAMPLE_THICKNESS = 4


# =========================
# APPLY (PURE GEOMETRY)
# =========================

def detect_progression_percent(img):

    runtime_height, runtime_width, _ = img.shape

    # --- Hard validation ---
    height_ratio = abs(runtime_height - EXPECTED_PROBE_HEIGHT) / EXPECTED_PROBE_HEIGHT

    if height_ratio > MAX_HEIGHT_DRIFT_RATIO:
        raise RuntimeError(
            f"Height drift too large. Expected {EXPECTED_PROBE_HEIGHT}, got {runtime_height}"
        )

    # --- Deterministic scaling ---
    scale = runtime_height / EXPECTED_PROBE_HEIGHT
    anchor_y = int(EXPECTED_ANCHOR_Y * scale)

    # --- Extract bar slice ---
    y1 = anchor_y - BAR_SAMPLE_THICKNESS // 2
    y2 = anchor_y + BAR_SAMPLE_THICKNESS // 2

    bar_slice = img[y1:y2, BAR_X_START:BAR_X_END]

    hsv = cv2.cvtColor(bar_slice, cv2.COLOR_BGR2HSV)

    # Detect non-white pixels
    saturation = hsv[:, :, 1]
    mask = saturation > 30

    filled_pixels = np.sum(mask)
    total_pixels = mask.size

    progression_percent = int((filled_pixels / total_pixels) * 100)

    return progression_percent


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

        # Deterministic: only process first page
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
