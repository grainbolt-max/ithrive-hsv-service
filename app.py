from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np
import cv2
import os

app = Flask(__name__)

ENGINE_NAME = "vMeasureRows"
API_KEY = "ithrive_secure_2026_key"


# ============================================================
# PDF IMAGE EXTRACTION
# ============================================================

def extract_panel_images(pdf_bytes):
    """
    Extract disease panel images from page 2.
    Assumes:
        image 0 = homeostasis
        image 1 = disease rows 1–12
        image 2 = disease rows 13–24
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if len(doc) < 2:
        raise Exception("PDF must contain at least 2 pages")

    page = doc[1]  # Page 2 (index 1)
    image_list = page.get_images(full=True)

    if len(image_list) < 3:
        raise Exception("Expected at least 3 images on page 2")

    panels = []

    for img_index in [1, 2]:
        xref = image_list[img_index][0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise Exception("Failed to decode panel image")

        panels.append(img)

    return panels


# ============================================================
# ROW MEASUREMENT ENGINE
# ============================================================

def measure_row_spacing(panel_img):
    """
    Detect horizontal divider lines using Sobel-Y
    and compute row spacing statistics.
    """

    panel_height = panel_img.shape[0]

    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    sobel_y = np.uint8(sobel_y)

    row_energy = np.sum(sobel_y, axis=1)

    if np.max(row_energy) == 0:
        return {
            "panel_height": int(panel_height),
            "detected_divider_count": 0,
            "min_row_height": None,
            "max_row_height": None,
            "mean_row_height": None,
            "std_dev_row_height": None,
            "all_row_heights": []
        }

    row_energy = row_energy / np.max(row_energy)

    divider_indices = np.where(row_energy > 0.4)[0]

    if len(divider_indices) == 0:
        return {
            "panel_height": int(panel_height),
            "detected_divider_count": 0,
            "min_row_height": None,
            "max_row_height": None,
            "mean_row_height": None,
            "std_dev_row_height": None,
            "all_row_heights": []
        }

    lines = []
    current_group = [divider_indices[0]]

    for idx in divider_indices[1:]:
        if idx - current_group[-1] <= 2:
            current_group.append(idx)
        else:
            lines.append(int(np.mean(current_group)))
            current_group = [idx]

    lines.append(int(np.mean(current_group)))

    if len(lines) < 2:
        return {
            "panel_height": int(panel_height),
            "detected_divider_count": len(lines),
            "min_row_height": None,
            "max_row_height": None,
            "mean_row_height": None,
            "std_dev_row_height": None,
            "all_row_heights": []
        }

    distances = np.diff(lines)

    return {
        "panel_height": int(panel_height),
        "detected_divider_count": int(len(lines)),
        "min_row_height": int(np.min(distances)),
        "max_row_height": int(np.max(distances)),
        "mean_row_height": float(np.mean(distances)),
        "std_dev_row_height": float(np.std(distances)),
        "all_row_heights": distances.tolist()
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return f"{ENGINE_NAME} running"


# ✅ HEALTH ROUTE (ADDED — REQUIRED FOR DEPLOY VALIDATION)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/v1/detect-disease-bars", methods=["POST"])
def measure_rows():

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = request.files["file"].read()

    try:
        panels = extract_panel_images(pdf_bytes)

        panel1_stats = measure_row_spacing(panels[0])
        panel2_stats = measure_row_spacing(panels[1])

        return jsonify({
            "engine": ENGINE_NAME,
            "panel_1_measurements": panel1_stats,
            "panel_2_measurements": panel2_stats
        })

    except Exception as e:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": str(e)
        }), 500


# ============================================================
# SERVER ENTRY
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
