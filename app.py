from flask import Flask, request, send_file
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import io
import gc

ENGINE_NAME = "ithrive_overlay_debug"
ENGINE_VERSION = "4.0_visual_debug"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150

X_LEFT = 703
X_RIGHT = 708

DISEASE_COORDINATES = {
    "large_artery_stiffness": (688, 700),
    "peripheral_vessel": (713, 725),
    "blood_pressure_uncontrolled": (738, 750),
    "small_medium_artery_stiffness": (763, 775),
    "atherosclerosis": (788, 800),
    "ldl_cholesterol": (813, 825),
    "lv_hypertrophy": (838, 850),
    "metabolic_syndrome": (875, 888),
    "insulin_resistance": (900, 913),
    "beta_cell_function_decreased": (925, 938),
    "blood_glucose_uncontrolled": (950, 963),
    "tissue_inflammatory_process": (975, 988),
    "hypothyroidism": (1145, 1160),
    "hyperthyroidism": (1170, 1183),
    "hepatic_fibrosis": (1195, 1210),
    "chronic_hepatitis": (1215, 1228),
    "prostate_cancer": (1238, 1250),
    "respiratory_disorders": (1263, 1275),
    "kidney_function_disorders": (1288, 1300),
    "digestive_disorders": (1313, 1325),
    "major_depression": (1360, 1373),
    "adhd_children_learning": (1380, 1393),
    "cerebral_dopamine_decreased": (1408, 1420),
    "cerebral_serotonin_decreased": (1425, 1438),
}

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

    # Draw vertical X window
    cv2.line(page, (X_LEFT, 0), (X_LEFT, height), (0, 0, 255), 2)
    cv2.line(page, (X_RIGHT, 0), (X_RIGHT, height), (0, 0, 255), 2)

    cv2.putText(page, f"X_LEFT={X_LEFT}", (X_LEFT+5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # Draw each Y band
    for name, (y1, y2) in DISEASE_COORDINATES.items():

        cv2.rectangle(page,
                      (X_LEFT, y1),
                      (X_RIGHT, y2),
                      (255, 0, 0),
                      2)

        cv2.putText(page,
                    f"{name} {y1}-{y2}",
                    (X_RIGHT + 10, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1)

    is_success, buffer = cv2.imencode(".png", page)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
