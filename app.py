from flask import Flask, request, jsonify
import numpy as np
from pdf2image import convert_from_bytes
import cv2

app = Flask(__name__)

# ==================================================
# DECLARE (FINAL THICKNESS-RANKED ENGINE)
# ==================================================

ENGINE_NAME = "v71_full_24_disease_mapping_thickness_ranked"
API_KEY = "ithrive_secure_2026_key"

DPI_LOCK = 200
EXPECTED_PROBE_HEIGHT = 2200
MAX_HEIGHT_DRIFT_RATIO = 0.03

VALUE_NONWHITE_THRESHOLD = 245
BAR_MIN_WIDTH = 700
BAR_MIN_HEIGHT = 8      # lower baseline — ranking handles filtering
VERTICAL_SCAN_STEP = 2

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
# APPLY
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


def detect_all_horizontal_bands(img):

    runtime_height, _, _ = img.shape

    height_ratio = abs(runtime_height - EXPECTED_PROBE_HEIGHT) / EXPECTED_PROBE_HEIGHT
    if height_ratio > MAX_HEIGHT_DRIFT_RATIO:
        raise RuntimeError(
            f"Height drift too large. Expected {EXPECTED_PROBE_HEIGHT}, got {runtime_height}"
        )

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    bands = []
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
                    bands.append({
                        "center": (y_top + y_bottom) // 2,
                        "height": band_height
                    })

                    y = y_bottom + 10
                    continue

        y += VERTICAL_SCAN_STEP

    return bands


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


def select_top_12_by_thickness(bands):

    if len(bands) < 12:
        raise RuntimeError(f"Expected at least 12 bands, detected {len(bands)}")

    # Sort by height descending
    bands_sorted = sorted(bands, key=lambda x: x["height"], reverse=True)

    # Take top 12 thickest
    top_12 = bands_sorted[:12]

    # Restore vertical order
    top_12_sorted_vertical = sorted(top_12, key=lambda x: x["center"])

    return [b["center"] for b in top_12_sorted_vertical]


def detect_all_24_diseases(pages):

    results = {}

    # ---------- PAGE 1 ----------
    page1_img = np.array(pages[0])
    page1_bands = detect_all_horizontal_bands(page1_img)
    page1_centers = select_top_12_by_thickness(page1_bands)

    for disease, y_center in zip(PAGE_1_DISEASES, page1_centers):
        percent = compute_bar_fill(page1_img, y_center)
        results[disease] = {
            "progression_percent": percent,
            "risk_label": "mild" if percent > 0 else "none",
            "source": ENGINE_NAME
        }

    # ---------- PAGE 2 ----------
    page2_img = np.array(pages[1])
    page2_bands = detect_all_horizontal_bands(page2_img)
    page2_centers = select_top_12_by_thickness(page2_bands)

    # Reverse for correct logical order
    page2_centers_reversed = list(reversed(page2_centers))

    for disease, y_center in zip(PAGE_2_DISEASES, page2_centers_reversed):
        percent = compute_bar_fill(page2_img, y_center)
        results[disease] = {
            "progression_percent": percent,
            "risk_label": "mild" if percent > 0 else "none",
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

        if len(pages) < 2:
            raise RuntimeError("PDF does not contain required pages")

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
