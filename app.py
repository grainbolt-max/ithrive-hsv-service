from flask import Flask, request, jsonify
import numpy as np
from pdf2image import convert_from_bytes
import cv2

app = Flask(__name__)

# ==================================================
# DECLARE (FINAL PRODUCTION ENGINE)
# ==================================================

ENGINE_NAME = "v76_full_24_color_detection_production"
API_KEY = "ithrive_secure_2026_key"

DPI_LOCK = 200
EXPECTED_PROBE_HEIGHT = 2200
MAX_HEIGHT_DRIFT_RATIO = 0.03

VALUE_NONWHITE_THRESHOLD = 245
BAR_MIN_WIDTH = 700
BAR_MIN_HEIGHT = 12
VERTICAL_SCAN_STEP = 2

# Hue thresholds calibrated from real data
RED_MAX = 8
ORANGE_MAX = 20
YELLOW_MAX = 40

SATURATION_THRESHOLD = 60  # Detect stripe pixels only

PAGE_1_DISEASES = [
    "large_artery_stiffness",
    "peripheral_vessel",
    "blood_pressure_uncontrolled",
    "small_medium_artery_stiffness",
    "atherosclerosis",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "metabolic_syndrome",
    "insulin_resistance",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "tissue_inflammatory_process"
]

PAGE_2_DISEASES = [
    "hypothyroidism",
    "hyperthyroidism",
    "hepatic_fibrosis",
    "chronic_hepatitis",
    "prostate_cancer",
    "respiratory_disorders",
    "kidney_function_disorders",
    "digestive_disorders",
    "major_depression",
    "adhd_children_learning",
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased"
]


# ==================================================
# DETECTION (STABLE V67 LOGIC)
# ==================================================

def measure_vertical_band(value_channel, start_y, x_start, x_end, height):
    y_top = start_y
    y_bottom = start_y

    while y_top > 0:
        row = value_channel[y_top, x_start:x_end]
        if np.mean(row) < VALUE_NONWHITE_THRESHOLD:
            y_top -= 1
        else:
            break

    while y_bottom < height - 1:
        row = value_channel[y_bottom, x_start:x_end]
        if np.mean(row) < VALUE_NONWHITE_THRESHOLD:
            y_bottom += 1
        else:
            break

    return y_top, y_bottom


def detect_all_bars_on_page(img):

    runtime_height, _, _ = img.shape

    height_ratio = abs(runtime_height - EXPECTED_PROBE_HEIGHT) / EXPECTED_PROBE_HEIGHT
    if height_ratio > MAX_HEIGHT_DRIFT_RATIO:
        raise RuntimeError(
            f"Height drift too large. Expected {EXPECTED_PROBE_HEIGHT}, got {runtime_height}"
        )

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

    return sorted(detected)


# ==================================================
# COLOR CLASSIFICATION (STRIPE-BASED)
# ==================================================

def classify_color_from_bar(img, y_center):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    row = value[y_center, :]
    col_mask = row < VALUE_NONWHITE_THRESHOLD
    nonzero = np.where(col_mask)[0]

    if len(nonzero) == 0:
        return "none"

    x_start = nonzero[0]

    # Wider sampling window to capture stripes
    y1 = max(0, y_center - 4)
    y2 = min(img.shape[0], y_center + 4)
    x1 = x_start + 5
    x2 = min(img.shape[1], x_start + 80)

    sample = hsv[y1:y2, x1:x2]

    sat = sample[:, :, 1]
    hue = sample[:, :, 0]

    # Only keep colored stripe pixels
    mask = sat > SATURATION_THRESHOLD

    if np.sum(mask) == 0:
        return "none"

    dominant_hue = float(np.mean(hue[mask]))

    # Severe (Red)
    if dominant_hue <= RED_MAX or dominant_hue >= 170:
        return "severe"

    # Moderate (Orange)
    if RED_MAX < dominant_hue <= ORANGE_MAX:
        return "moderate"

    # Mild (Yellow)
    if ORANGE_MAX < dominant_hue <= YELLOW_MAX:
        return "mild"

    return "none"


# ==================================================
# MAIN 24-DISEASE MAPPING
# ==================================================

def detect_all_24_diseases(pages):

    global_bars = []

    for page_index in [0, 1]:
        img = np.array(pages[page_index])
        bars = detect_all_bars_on_page(img)
        for y_center in bars:
            global_bars.append((page_index, y_center))

    if len(global_bars) != 24:
        raise RuntimeError(f"Expected 24 total bars, detected {len(global_bars)}")

    results = {}

    page1_bars = global_bars[:12]
    page2_bars = global_bars[12:]
    page2_bars_reversed = list(reversed(page2_bars))

    for disease, (page_index, y_center) in zip(PAGE_1_DISEASES, page1_bars):
        img = np.array(pages[page_index])
        risk = classify_color_from_bar(img, y_center)
        results[disease] = {
            "risk_label": risk,
            "source": ENGINE_NAME
        }

    for disease, (page_index, y_center) in zip(PAGE_2_DISEASES, page2_bars_reversed):
        img = np.array(pages[page_index])
        risk = classify_color_from_bar(img, y_center)
        results[disease] = {
            "risk_label": risk,
            "source": ENGINE_NAME
        }

    return results


# ==================================================
# OUTPUT
# ==================================================

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

        results = detect_all_24_diseases(pages)

        return jsonify({
            "engine": ENGINE_NAME,
            "pages_found": len(pages),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
