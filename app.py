from flask import Flask, request, send_file
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import json
import gc

ENGINE_NAME = "ithrive_overlay_debug"
ENGINE_VERSION = "1.5.0_png_output"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150

@app.route("/v1/overlay", methods=["POST"])
def generate_overlay():

    if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
        return {"error": "Unauthorized"}, 401

    if "file" not in request.files:
        return {"error": "No file provided"}, 400

    pdf_bytes = request.files["file"].read()

    pages = convert_from_bytes(
        pdf_bytes,
        dpi=RENDER_DPI,
        first_page=2,
        last_page=2
    )

    page_image = np.array(pages[0])
    del pages
    gc.collect()

    image_height, image_width = page_image.shape[:2]

    overlay = page_image.copy()

    # Draw vertical center line so we see true page center
    center_x = image_width // 2
    cv2.line(overlay, (center_x, 0), (center_x, image_height), (0, 0, 255), 2)

    # Save file
    output_path = "/tmp/overlay.png"
    cv2.imwrite(output_path, overlay)

    del page_image
    del overlay
    gc.collect()

    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
