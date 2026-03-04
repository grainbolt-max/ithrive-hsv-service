from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import tempfile

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"


BASE_LAYOUT = {

    # PAGE 2 – CARDIO / DIABETES

    "large_artery_stiffness": {"x": 860, "y": 750, "w": 520, "h": 42},
    "peripheral_vessel": {"x": 860, "y": 792, "w": 520, "h": 42},
    "blood_pressure_uncontrolled": {"x": 860, "y": 834, "w": 520, "h": 42},
    "small_medium_artery": {"x": 860, "y": 876, "w": 520, "h": 42},
    "atherosclerosis": {"x": 860, "y": 918, "w": 520, "h": 42},
    "ldl_cholesterol": {"x": 860, "y": 960, "w": 520, "h": 42},
    "lv_hypertrophy": {"x": 860, "y": 1002, "w": 520, "h": 42},

    "metabolic_syndrome": {"x": 860, "y": 1080, "w": 520, "h": 42},
    "insulin_resistance": {"x": 860, "y": 1122, "w": 520, "h": 42},
    "beta_cell_function": {"x": 860, "y": 1164, "w": 520, "h": 42},
    "blood_glucose": {"x": 860, "y": 1206, "w": 520, "h": 42},
    "tissue_inflammation": {"x": 860, "y": 1248, "w": 520, "h": 42},

    # PAGE 3 – MISC

    "hypothyroidism": {"x": 860, "y": 520, "w": 520, "h": 42},
    "hyperthyroidism": {"x": 860, "y": 562, "w": 520, "h": 42},
    "hepatic_fibrosis": {"x": 860, "y": 604, "w": 520, "h": 42},
    "chronic_hepatitis": {"x": 860, "y": 646, "w": 520, "h": 42},

    "respiratory_disorders": {"x": 860, "y": 726, "w": 520, "h": 42},
    "kidney_function": {"x": 860, "y": 768, "w": 520, "h": 42},
    "digestive_disorders": {"x": 860, "y": 810, "w": 520, "h": 42},

    "major_depression": {"x": 860, "y": 920, "w": 520, "h": 42},
    "adhd_learning": {"x": 860, "y": 962, "w": 520, "h": 42},
    "dopamine_decrease": {"x": 860, "y": 1004, "w": 520, "h": 42},
    "serotonin_decrease": {"x": 860, "y": 1046, "w": 520, "h": 42},
}


def check_auth(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"


@app.route("/")
def home():
    return jsonify({"status": "ITHRIVE HSV Service Running"})


@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()

    pages = convert_from_bytes(file_bytes, dpi=300)

    page2 = np.array(pages[1])
    page3 = np.array(pages[2])

    combined = np.vstack((page2, page3))
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

    pages = convert_from_bytes(file_bytes, dpi=300)

    page2 = np.array(pages[1])
    page3 = np.array(pages[2])

    combined = np.vstack((page2, page3))

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
