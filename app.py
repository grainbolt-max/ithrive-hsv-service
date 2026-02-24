import numpy as np
import cv2
from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"
ENGINE_NAME = "hsv_v41_auto_track_detection"

# ---------- Yellow Calibration ----------
YELLOW_HUE_MIN = 18
YELLOW_HUE_MAX = 38
SAT_MIN = 45
VAL_MIN = 105

GRAY_DELTA = 15

LEFT_BOUND = 450
RIGHT_BOUND = 1200

DISEASE_ROWS = [
    "adhd_children_learning",
    "atherosclerosis",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "blood_pressure_uncontrolled",
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased",
    "chronic_hepatitis",
    "digestive_disorders",
    "hepatic_fibrosis",
    "hyperthyroidism",
    "hypothyroidism",
    "insulin_resistance",
    "kidney_function_disorders",
    "large_artery_stiffness",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "major_depression",
    "metabolic_syndrome",
    "peripheral_vessel",
    "prostate_cancer",
    "respiratory_disorders",
    "small_medium_artery_stiffness",
    "tissue_inflammatory_process",
]

# --------------------------------------------------------

def is_gray(b, g, r):
    return (
        abs(int(r) - int(g)) < GRAY_DELTA and
        abs(int(r) - int(b)) < GRAY_DELTA and
        abs(int(g) - int(b)) < GRAY_DELTA
    )

def detect_tracks(image):
    h, w, _ = image.shape
    gray_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(LEFT_BOUND, min(RIGHT_BOUND, w)):
            b, g, r = image[y, x]
            if is_gray(b, g, r):
                gray_mask[y, x] = 255

    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tracks = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        if cw > 500 and 15 < ch < 40:
            tracks.append((x, y, cw, ch))

    tracks = sorted(tracks, key=lambda t: t[1])
    return tracks

def isolate_yellow_span(track_img):
    hsv = cv2.cvtColor(track_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    yellow_mask = (
        (h >= YELLOW_HUE_MIN) &
        (h <= YELLOW_HUE_MAX) &
        (s >= SAT_MIN) &
        (v >= VAL_MIN)
    )

    cols = yellow_mask.any(axis=0)

    if not np.any(cols):
        return 0

    span_pixels = np.sum(cols)
    total_pixels = yellow_mask.shape[1]

    return int((span_pixels / total_pixels) * 100)

def classify(percent):
    if percent >= 75:
        return "severe"
    if percent >= 50:
        return "moderate"
    if percent >= 25:
        return "mild"
    if percent > 0:
        return "normal"
    return "none"

@app.route("/")
def health():
    return "HSV Preprocess Service Running v41"

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        pdf_bytes = request.files["file"].read()
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except:
        return jsonify({"error": "PDF conversion failed"}), 500

    results = {}
    disease_index = 0

    for page in pages:
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        tracks = detect_tracks(img)

        for track in tracks:
            if disease_index >= len(DISEASE_ROWS):
                break

            x, y, w, h = track
            track_img = img[y:y+h, x:x+w]

            percent = isolate_yellow_span(track_img)

            disease_name = DISEASE_ROWS[disease_index]

            results[disease_name] = {
                "progression_percent": percent,
                "risk_label": classify(percent),
                "source": ENGINE_NAME,
            }

            disease_index += 1

    return jsonify({
        "engine": ENGINE_NAME,
        "pages_found": len(pages),
        "results": results
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
