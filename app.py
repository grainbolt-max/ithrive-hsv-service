from flask import Flask, request, send_file
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import io
import json
import gc

ENGINE_NAME = "ithrive_y_debug"
ENGINE_VERSION = "3.0_y_overlay"

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

    layout_json = request.form.get("layout_profile")
    if not layout_json:
        return "layout_profile required", 400

    layout = json.loads(layout_json)

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

    # ============================
    # GRID FOR REFERENCE
    # ============================

    for y in range(0, height, 100):
        cv2.line(page, (0, y), (width, y), (0, 255, 0), 1)
        cv2.putText(page, str(y), (5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # ============================
    # DRAW ALL Y ROWS
    # ============================

    for panel_name in ["panel_1", "panel_2"]:
        rows = layout["panels"][panel_name]["rows"]

        for key, row in rows.items():
            y_top = int(row["y_top"])
            y_bottom = int(row["y_bottom"])

            # TOP line (RED)
            cv2.line(page, (0, y_top), (width, y_top), (0, 0, 255), 2)

            # BOTTOM line (BLUE)
            cv2.line(page, (0, y_bottom), (width, y_bottom), (255, 0, 0), 2)

            cv2.putText(page,
                        f"{key}",
                        (20, y_top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1)

    # ============================

    is_success, buffer = cv2.imencode(".png", page)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
