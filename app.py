from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import hashlib
import base64
import io
import os

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"


# ----------------------------------------------------
# AUTH
# ----------------------------------------------------

def require_auth(req):
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split("Bearer ")[1].strip()
    return token == API_KEY


# ----------------------------------------------------
# TEMPLATE ALIGNMENT
# ----------------------------------------------------

def register_to_template(page):

    template_path = "page2_template.png"

    if not os.path.exists(template_path):
        return page

    template = cv2.imread(template_path)

    gray1 = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return page

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return page

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches[:50]]
    ).reshape(-1,1,2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches[:50]]
    ).reshape(-1,1,2)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is None:
        return page

    aligned = cv2.warpAffine(
        page,
        M,
        (template.shape[1], template.shape[0])
    )

    return aligned


# ----------------------------------------------------
# COLOR DETECTION
# ----------------------------------------------------

def detect_color_presence(region):

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    lower = np.array([70,50,50])
    upper = np.array([170,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    return cv2.countNonZero(mask) > 200


# ----------------------------------------------------
# BAR COLUMN (SET BY CALIBRATION)
# ----------------------------------------------------

BAR_X = 1120
BAR_WIDTH = 340

BASE_LAYOUT = {

    "large_artery_stiffness": {"y":750,"h":42},
    "peripheral_vessel": {"y":792,"h":42},
    "blood_pressure_uncontrolled": {"y":834,"h":42},
    "small_medium_artery": {"y":876,"h":42},
    "atherosclerosis": {"y":918,"h":42},
    "ldl_cholesterol": {"y":960,"h":42},
    "lv_hypertrophy": {"y":1002,"h":42},

    "metabolic_syndrome": {"y":1080,"h":42},
    "insulin_resistance": {"y":1122,"h":42},
    "beta_cell_function": {"y":1164,"h":42},
    "blood_glucose": {"y":1206,"h":42},
    "tissue_inflammation": {"y":1248,"h":42},

    "hypothyroidism": {"y":520,"h":42},
    "hyperthyroidism": {"y":562,"h":42},
    "hepatic_fibrosis": {"y":604,"h":42},
    "chronic_hepatitis": {"y":646,"h":42},

    "respiratory_disorders": {"y":726,"h":42},
    "kidney_function": {"y":768,"h":42},
    "digestive_disorders": {"y":810,"h":42},

    "major_depression": {"y":920,"h":42},
    "adhd_learning": {"y":962,"h":42},
    "dopamine_decrease": {"y":1004,"h":42},
    "serotonin_decrease": {"y":1046,"h":42},
}


# ----------------------------------------------------
# PDF METADATA
# ----------------------------------------------------

@app.route("/v1/pdf-metadata", methods=["POST"])
def pdf_metadata():

    if not require_auth(request):
        return jsonify({"error":"Unauthorized"}),401

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    first_page = np.array(images[0])

    h,w = first_page.shape[:2]

    small = cv2.resize(first_page,(200,200))
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)

    sha = hashlib.sha256(gray.tobytes()).digest()

    pixel_hash = base64.b64encode(sha).decode("utf-8")

    return jsonify({
        "page_width":w,
        "page_height":h,
        "file_size":len(pdf_bytes),
        "page_count":len(images),
        "pixel_hash_b64":pixel_hash
    })


# ----------------------------------------------------
# DISEASE DETECTION
# ----------------------------------------------------

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    if not require_auth(request):
        return jsonify({"error":"Unauthorized"}),401

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(images[1])

    page = register_to_template(page)

    results = {}

    for disease,coords in BASE_LAYOUT.items():

        y = coords["y"]
        h = coords["h"]

        region = page[y:y+h, BAR_X:BAR_X+BAR_WIDTH]

        if region.size == 0:
            continue

        has_color = detect_color_presence(region)

        results[disease] = "Moderate" if has_color else "None/Low"

    return jsonify({
        "engine":"ithrive_hsv_template_locked",
        "results":results
    })


# ----------------------------------------------------
# DEBUG OVERLAY
# ----------------------------------------------------

@app.route("/v1/debug-overlay", methods=["POST"])
def debug_overlay():

    if not require_auth(request):
        return jsonify({"error":"Unauthorized"}),401

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(images[1])

    page = register_to_template(page)

    overlay = page.copy()

    for disease,coords in BASE_LAYOUT.items():

        y = coords["y"]
        h = coords["h"]

        cv2.rectangle(
            overlay,
            (BAR_X,y),
            (BAR_X+BAR_WIDTH,y+h),
            (0,0,255),
            2
        )

    _,buffer = cv2.imencode(".png",overlay)

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype="image/png"
    )


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status":"ITHRIVE HSV Service Running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
