from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import json
import gc
import base64

ENGINE_NAME = "ithrive_color_engine_page2_coordinate_lock_v1_PRODUCTION"
ENGINE_VERSION = "1.4.0_overlay_debug"

API_KEY = os.environ.get("ITHRIVE_API_KEY")
if not API_KEY:
    raise RuntimeError("ITHRIVE_API_KEY not set")

app = Flask(__name__)

RENDER_DPI = 150

PANEL_1_KEYS = [
    "large_artery_stiffness",
    "peripheral_vessel",
    "blood_pressure_uncontrolled",
    "small_medium_artery_stiffness",
    "atherosclerosis",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "metabolic_syndrome",
    "insulin_resistance",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "tissue_inflammatory_process",
]

PANEL_2_KEYS = [
    "hypothyroidism",
    "hyperthyroidism",
    "hepatic_fibrosis",
    "chronic_hepatitis",
    "prostate_cancer",
    "respiratory_disorders",
    "kidney_function_disorders",
    "digestive_disorders",
    "major_depression",
    "adhd_children_learning",
    "cerebral_dopamine_decreased",
    "cerebral_serotonin_decreased",
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "engine": ENGINE_NAME,
        "version": ENGINE_VERSION
    })

@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    if request.headers.get("Authorization", "") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    layout_json = request.form.get("layout_profile")
    if not layout_json:
        return jsonify({"error": "layout_mismatch"}), 400

    layout = json.loads(layout_json)

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

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

    x_left = int(layout["x_left"])
    x_right = int(layout["x_right"])

    # Draw vertical sampling band
    cv2.line(overlay, (x_left, 0), (x_left, image_height), (0, 0, 255), 2)
    cv2.line(overlay, (x_right, 0), (x_right, image_height), (0, 0, 255), 2)

    # Draw disease row boxes
    for panel_name, keys in [
        ("panel_1", PANEL_1_KEYS),
        ("panel_2", PANEL_2_KEYS),
    ]:
        rows = layout["panels"][panel_name]["rows"]

        for key in keys:
            y_top = int(rows[key]["y_top"])
            y_bottom = int(rows[key]["y_bottom"])

            cv2.rectangle(
                overlay,
                (x_left, y_top),
                (x_right, y_bottom),
                (0, 255, 0),
                2
            )

    _, buffer = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    del page_image
    del overlay
    gc.collect()

    return jsonify({
        "engine": ENGINE_NAME,
        "dpi": RENDER_DPI,
        "width": image_width,
        "height": image_height,
        "overlay_image_base64": overlay_base64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
