from flask import Flask, request, send_file
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import io
import gc

ENGINE_NAME = "ithrive_y_visual_only"
ENGINE_VERSION = "3.1_grid_only"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150

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

    # Draw horizontal grid every 50px
    for y in range(0, height, 50):
        color = (0,255,0) if y % 100 == 0 else (200,200,200)
        cv2.line(page, (0,y), (width,y), color, 1)

        if y % 100 == 0:
            cv2.putText(page, str(y),
                        (10, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        1)

    # Draw vertical grid every 100px
    for x in range(0, width, 100):
        cv2.line(page, (x,0), (x,height), (0,0,255), 1)
        cv2.putText(page, str(x),
                    (x+5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    1)

    is_success, buffer = cv2.imencode(".png", page)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
