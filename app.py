from flask import Flask, request, jsonify, send_file
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import io
import hashlib
import base64
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
# REMOVE SCANNER MARGIN
# ----------------------------------------------------

def autocrop_page(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    for y in range(h):

        row = gray[y:y+1, :]

        dark_ratio = np.mean(row < 230)

        if dark_ratio > 0.02:
            top = max(0, y - 20)
            return img[top:h, :]

    return img


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
# FIND ROWS AUTOMATICALLY
# ----------------------------------------------------

def detect_rows(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    horizontal = cv2.reduce(edges, 1, cv2.REDUCE_AVG)

    rows = []

    threshold = 10

    start = None

    for y, val in enumerate(horizontal):

        if val > threshold and start is None:
            start = y

        elif val <= threshold and start is not None:

            height = y - start

            if height > 25 and height < 80:
                rows.append((start, height))

            start = None

    return rows


# ----------------------------------------------------
# HSV COLOR DETECTION
# ----------------------------------------------------

def detect_bar_color(region):

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    lower = np.array([70,50,50])
    upper = np.array([170,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    pixels = cv2.countNonZero(mask)

    return pixels > 200


# ----------------------------------------------------
# BAR COLUMN (CALIBRATED)
# ----------------------------------------------------

BAR_X = 1120
BAR_WIDTH = 340


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
# DETECT DISEASE BARS
# ----------------------------------------------------

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    if not require_auth(request):
        return jsonify({"error":"Unauthorized"}),401

    pdf_bytes = request.files["file"].read()

    images = convert_from_bytes(pdf_bytes, dpi=200)

    page = np.array(images[1])

    page = autocrop_page(page)

    page = register_to_template(page)

    rows = detect_rows(page)

    results = {}

    for i,(y,h) in enumerate(rows):

        region = page[y:y+h, BAR_X:BAR_X+BAR_WIDTH]

        if region.size == 0:
            continue

        has_color = detect_bar_color(region)

        results[f"row_{i}"] = "Moderate" if has_color else "None/Low"

    return jsonify({
        "engine":"ithrive_row_detection_engine",
        "rows_detected":len(rows),
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

    page = autocrop_page(page)

    page = register_to_template(page)

    rows = detect_rows(page)

    overlay = page.copy()

    for y,h in rows:

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


# ----------------------------------------------------
# HEALTH
# ----------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status":"ITHRIVE HSV Service Running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
