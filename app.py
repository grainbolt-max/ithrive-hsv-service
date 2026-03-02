from flask import Flask, request, send_file
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import io
import gc

ENGINE_NAME = "ithrive_visual_debug"
ENGINE_VERSION = "2.0_visual_overlay"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150

# CHANGE THIS TO TEST ANY X POSITION
TEST_X = 905   # <-- You can change this to 745, 706, etc.

@app.route("/v1/overlay", methods=["POST"])
def overlay():

    if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
        return "Unauthorized", 401

    if "file" not in request.files:
        return "No file", 400

    pdf_bytes = request.files["file"].read()

    pages = convert_from_bytes(
        pdf_bytes,
        dpi=RENDER_DPI,
        first_page=2,
        last_page=2
    )

    page = np.array(pages[0])
    del pages
    gc.collect()

    height, width = page.shape[:2]

    # ======================================
    # DRAW GRID
    # ======================================

    for x in range(0, width, 100):
        cv2.line(page, (x, 0), (x, height), (0, 0, 255), 1)
        cv2.putText(page, str(x), (x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

    for y in range(0, height, 100):
        cv2.line(page, (0, y), (width, y), (0, 255, 0), 1)
        cv2.putText(page, str(y), (5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # ======================================
    # DRAW TEST SAMPLING BAND
    # ======================================

    cv2.line(page, (TEST_X, 0), (TEST_X, height), (255, 0, 0), 3)
    cv2.putText(page, f"TEST_X = {TEST_X}",
                (TEST_X + 10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

    # ======================================

    is_success, buffer = cv2.imencode(".png", page)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
