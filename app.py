from flask import Flask, request, jsonify
import numpy as np
import cv2
from pdf2image import convert_from_bytes

from parser.anchors import detect_all_anchors
from parser.rows import detect_row_lines
from parser.extract import extract_disease_scores

app = Flask(__name__)

API_KEY = "ithrive_secure_2026_key"


@app.route("/parse-report", methods=["POST"])
def parse_report():

    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]

    pages = convert_from_bytes(pdf_file.read())

    page = np.array(pages[1])

    img = cv2.cvtColor(page, cv2.COLOR_RGB2BGR)

    anchors = detect_all_anchors(img)

    rows = detect_row_lines(
        img,
        anchors["header_y"],
        anchors["table_bottom_y"]
    )

    scores = extract_disease_scores(img, rows, anchors)

    return jsonify({
        "engine": "ithrive_disease_parser_v1",
        "anchors": anchors,
        "disease_scores": scores
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)