from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import tempfile

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"

# SINGLE CONTROL FOR BAR POSITION
BAR_X = 300

BASE_LAYOUT = {

    # PAGE 2 – CARDIO / DIABETES

    "large_artery_stiffness": {"x": BAR_X, "y": 110, "w": 520, "h": 42},
    "peripheral_vessel": {"x": BAR_X, "y": 152, "w": 520, "h": 42},
    "blood_pressure_uncontrolled": {"x": BAR_X, "y": 194, "w": 520, "h": 42},
    "small_medium_artery": {"x": BAR_X, "y": 236, "w": 520, "h": 42},
    "atherosclerosis": {"x": BAR_X, "y": 278, "w": 520, "h": 42},
    "ldl_cholesterol": {"x": BAR_X, "y": 320, "w": 520, "h": 42},
    "lv_hypertrophy": {"x": BAR_X, "y": 362, "w": 520, "h": 42},

    "metabolic_syndrome": {"x": BAR_X, "y": 440, "w": 520, "h": 42},
    "insulin_resistance": {"x": BAR_X, "y": 482, "w": 520, "h": 42},
    "beta_cell_function": {"x": BAR_X, "y": 524, "w": 520, "h": 42},
    "blood_glucose": {"x": BAR_X, "y": 566, "w": 520, "h": 42},
    "tissue_inflammation": {"x": BAR_X, "y": 608, "w": 520, "h": 42},

    # PAGE 3 – MISC DISEASES

    "hypothyroidism": {"x": BAR_X, "y": 870, "w": 520, "h": 42},
    "hyperthyroidism": {"x": BAR_X, "y": 912, "w": 520, "h": 42},
    "hepatic_fibrosis": {"x": BAR_X, "y": 954, "w": 520, "h": 42},
    "chronic_hepatitis": {"x": BAR_X, "y": 996, "w": 520, "h": 42},

    "respiratory_disorders": {"x": BAR_X, "y": 1076, "w": 520, "h": 42},
    "kidney_function": {"x": BAR_X, "y": 1118, "w": 520, "h": 42},
    "digestive_disorders": {"x": BAR_X, "y": 1160, "w": 520, "h": 42},

    "major_depression": {"x": BAR_X, "y": 1270, "w": 520, "h": 42},
    "adhd_learning": {"x": BAR_X, "y": 1312, "w": 520, "h": 42},
    "dopamine_decrease": {"x": BAR_X, "y": 1354, "w": 520, "h": 42},
    "serotonin_decrease": {"x": BAR_X, "y": 1396, "w": 520, "h": 42},
}


def check_auth(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"


@app.route("/")
def home():
    return jsonify({"status": "ITHRIVE HSV Service Running"})


def extract_pages(file_bytes):

    pages = convert_from_bytes(file_bytes, dpi=300)

    page2 = np.array(pages[1])
    page3 = np.array(pages[2])

    # crop analyzer section off the top
    page2 = page2[900:2500, :]
    page3 = page3[400:2000, :]

    combined = np.vstack((page2, page3))

    return combined


@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()

    combined = extract_pages(file_bytes)

    overlay = combined.copy()

    for name, box in BASE_LAYOUT.items():

        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp.name, overlay)

    return send_file(temp.name, mimetype="image/png")


@app.route("/v1/analyze", methods=["POST"])
def analyze():

    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()

    combined = extract_pages(file_bytes)

    results = {}

    for name, box in BASE_LAYOUT.items():

        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        crop = combined[y:y+h, x:x+w]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        ratio = np.sum(mask > 0) / (w * h)

        results[name] = round(float(ratio), 3)

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
