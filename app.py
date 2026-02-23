from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import cv2
import numpy as np
import os

app = Flask(__name__)

ENGINE_VERSION = "hsv_v26_overlay_span_locked"
API_KEY = "ithrive_secure_2026_key"

ROWS_PER_PAGE = 14
TOP_MARGIN_RATIO = 0.18
BOTTOM_MARGIN_RATIO = 0.08
LEFT_SAMPLE_RATIO = 0.15
RIGHT_SAMPLE_RATIO = 0.95

GRAY_LOW = np.array([0, 0, 40])
GRAY_HIGH = np.array([180, 40, 160])

YELLOW_LOW = np.array([15, 80, 150])
YELLOW_HIGH = np.array([40, 255, 255])

ORANGE_LOW = np.array([5, 120, 150])
ORANGE_HIGH = np.array([20, 255, 255])

RED_LOW_1 = np.array([0, 120, 150])
RED_HIGH_1 = np.array([10, 255, 255])
RED_LOW_2 = np.array([170, 120, 150])
RED_HIGH_2 = np.array([180, 255, 255])

COLOR_MAP = {
    "none": 20,
    "mild": 50,
    "moderate": 70,
    "severe": 85,
    "normal": 10
}

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
    "tissue_inflammatory_process",
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
    "cerebral_serotonin_decreased",
]


def classify_color(hsv_pixel):
    if cv2.inRange(hsv_pixel, YELLOW_LOW, YELLOW_HIGH):
        return "mild"
    if cv2.inRange(hsv_pixel, ORANGE_LOW, ORANGE_HIGH):
        return "moderate"
    if (cv2.inRange(hsv_pixel, RED_LOW_1, RED_HIGH_1) or
        cv2.inRange(hsv_pixel, RED_LOW_2, RED_HIGH_2)):
        return "severe"
    return "none"


def is_gray(hsv_pixel):
    return cv2.inRange(hsv_pixel, GRAY_LOW, GRAY_HIGH)


def process_row_overlay_span(row_img):
    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv.shape

    sample_y = height // 2
    left_bound = int(width * LEFT_SAMPLE_RATIO)
    right_bound = int(width * RIGHT_SAMPLE_RATIO)

    overlay_pixels = []

    for x in range(left_bound, right_bound):
        pixel = hsv[sample_y:sample_y+1, x:x+1]
        if not is_gray(pixel):
            overlay_pixels.append(x)

    if not overlay_pixels:
        return "none"

    span_left = min(overlay_pixels)
    span_right = max(overlay_pixels)
    center_x = (span_left + span_right) // 2
    center_pixel = hsv[sample_y:sample_y+1, center_x:center_x+1]

    return classify_color(center_pixel)


def render_page_to_image(page):
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


@app.route("/")
def home():
    return f"HSV Preprocess Service Running v26"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_bytes = request.files["file"].read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    results = {}
    target_pages = min(len(doc), 2)

    for page_index in range(target_pages):
        page = doc.load_page(page_index)
        img = render_page_to_image(page)

        height = img.shape[0]
        top = int(height * TOP_MARGIN_RATIO)
        bottom = int(height * (1 - BOTTOM_MARGIN_RATIO))
        content = img[top:bottom, :]

        row_height = content.shape[0] // ROWS_PER_PAGE

        for row_index in range(ROWS_PER_PAGE):

            if row_index in [0, 7]:
                continue

            row_top = row_index * row_height
            row_bottom = row_top + row_height
            row_img = content[row_top:row_bottom, :]

            if page_index == 0:
                disease_list = PAGE_1_DISEASES
                disease_index = row_index - 1
            else:
                disease_list = PAGE_2_DISEASES
                disease_index = row_index - 1

            if 0 <= disease_index < len(disease_list):
                key = disease_list[disease_index]
                color_label = process_row_overlay_span(row_img)
                progression = COLOR_MAP[color_label]

                results[key] = {
                    "progression_percent": progression,
                    "risk_label": color_label,
                    "source": ENGINE_VERSION
                }

    return jsonify({
        "engine": ENGINE_VERSION,
        "pages_found": target_pages,
        "results": results
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
