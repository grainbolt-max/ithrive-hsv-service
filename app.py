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
# TEMPLATE REGISTRATION
# Aligns incoming scans to a known reference
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
# TABLE ROW DETECTION
# Hospital-style dynamic layout detection
# ----------------------------------------------------

def detect_table_rows(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))

    horizontal = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rows = []

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        if w > 400:
            rows.append((x,y,w,h))

    rows = sorted(rows, key=lambda r: r[1])

    return rows


# ----------------------------------------------------
# COLOR BAR DETECTION
# ----------------------------------------------------

def detect_color_presence(region):

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    lower = np.array([70,50,50])
    upper = np.array([170,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    pixels = cv2.countNonZero(mask)

    return pixels > 200


# ----------------------------------------------------
# PDF METADATA
# ----------------------------------------------------

@app.route("/v1/pdf-metadata", methods=["POST"])
def pdf_metadata():

    if not require_auth(request):
        return jsonify({"error":"Unauthorized"}),401

    if "file" not in request.files:
        return jsonify({"error":"No file"}),400

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
# MAIN ENGINE
# ----------------------------------------------------

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    if not require_auth(request):
        return jsonify({"error":"Unauthorized"}),401

    if "file" not in request.files:
        return jsonify({"error":"No file"}),400

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(images[1])

    page = register_to_template(page)

    rows = detect_table_rows(page)

    results = {}

    idx = 0

    for x,y,w,h in rows:

        region = page[y:y+h, 1040:1560]

        if region.size == 0:
            continue

        has_color = detect_color_presence(region)

        label = f"row_{idx}"

        results[label] = "Moderate" if has_color else "None/Low"

        idx += 1

    return jsonify({
        "engine":"ithrive_hospital_engine_v1",
        "rows_detected":len(results),
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

    rows = detect_table_rows(page)

    overlay = page.copy()

    for x,y,w,h in rows:

        cv2.rectangle(
            overlay,
            (x,y),
            (x+w,y+h),
            (0,0,255),
            2
        )

    _,buffer = cv2.imencode(".png",overlay)

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype="image/png"
    )


# ----------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":"ITHRIVE HSV Service Running"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
