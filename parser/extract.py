from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import numpy as np
from pdf2image import convert_from_bytes
import os
import cv2

from parser.layout_registry import fingerprint_layout, register_layout
from parser.anchors import detect_all_anchors
from parser.rows import detect_rows
from parser.extract import extract_disease_scores
from parser.layout_normalizer import normalize_dpi
from parser.system_engine import compute_system_summary, compute_consultation_summary
from engine.pattern_engine import detect_patterns
from engine.protocol_engine import build_protocol
from engine.narrative_engine import generate_health_narrative

server = Flask(__name__, static_folder="static")
CORS(server)

ENGINE_NAME = "ithrive_disease_parser_v1"
API_KEY = "ithrive_secure_2026_key"

# ================================
# MANUAL SAMPLE STRIPE (ADJUST)
# ================================
SCORE_SAMPLE_LEFT = 750
SCORE_SAMPLE_RIGHT = 756


@server.route("/")
def root():
    return jsonify({
        "engine": ENGINE_NAME,
        "status": "ok"
    })


@server.route("/health")
def health():
    return jsonify({
        "engine": ENGINE_NAME,
        "status": "ok"
    })


@server.route("/docs")
def docs():
    return send_from_directory("static", "docs.html")


@server.route("/debug-crop", methods=["POST"])
def debug_crop():

    if "file" not in request.files:
        return "missing file", 400

    file = request.files["file"]
    pdf_bytes = file.read()

    pages = convert_from_bytes(pdf_bytes)

    if len(pages) < 2:
        return "missing disease page", 400

    page = np.array(pages[1])
    page, _ = normalize_dpi(page)

    anchors = detect_all_anchors(page)
    rows = detect_rows(page, anchors)

    debug_img = page.copy()

    # ===================================
    # DRAW ANCHOR POINTS
    # ===================================
    if isinstance(anchors, dict):
        for k, v in anchors.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                x, y = v
                cv2.circle(debug_img, (int(x), int(y)), 10, (255, 0, 0), -1)

    # ===================================
    # DRAW ROW BOXES
    # ===================================
    for r in rows:

        if isinstance(r, (list, tuple)) and len(r) == 4:
            x1, y1, x2, y2 = r

        elif isinstance(r, (int, float)):
            y = int(r)
            x1 = 0
            x2 = debug_img.shape[1]
            y1 = y - 5
            y2 = y + 5

        else:
            continue

        cv2.rectangle(
            debug_img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

    # ===================================
    # DRAW X GRID EVERY 25px
    # ===================================
    for x in range(0, debug_img.shape[1], 25):

        cv2.line(
            debug_img,
            (x, 0),
            (x, debug_img.shape[0]),
            (180, 180, 180),
            1
        )

        cv2.putText(
            debug_img,
            str(x),
            (x + 2, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1
        )

    # ===================================
    # DRAW SAMPLING STRIPE
    # ===================================
    cv2.rectangle(
        debug_img,
        (SCORE_SAMPLE_LEFT, 0),
        (SCORE_SAMPLE_RIGHT, debug_img.shape[0]),
        (255, 0, 255),
        3
    )

    success, buffer = cv2.imencode(".png", debug_img)

    if not success:
        return "image encoding failed", 500

    return Response(buffer.tobytes(), mimetype="image/png")


@server.route("/parse-report", methods=["POST"])
def parse_report():

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    pages = convert_from_bytes(pdf_bytes)

    if len(pages) < 2:
        return jsonify({"error": "report missing disease page"}), 400

    page = np.array(pages[1])
    page, _ = normalize_dpi(page)

    anchors = detect_all_anchors(page)
    rows = detect_rows(page, anchors)

    layout_hash = fingerprint_layout(page, anchors, rows)
    register_layout(layout_hash, anchors, rows)

    scores = extract_disease_scores(page, anchors, rows)
    patterns = detect_patterns(scores)
    protocol = build_protocol(patterns)

    system_summary = compute_system_summary(scores)
    consultation_summary = compute_consultation_summary(system_summary)

    narrative = generate_health_narrative(
        system_summary,
        consultation_summary,
        protocol
    )

    return jsonify({
        "engine": ENGINE_NAME,
        "layout_id": layout_hash,
        "system_summary": system_summary,
        "consultation_summary": consultation_summary,
        "protocol": protocol,
        "health_narrative": narrative,
        "disease_scores": scores
    })


if __name__ == "__main__":

    print("\nREGISTERED ROUTES:")
    for rule in server.url_map.iter_rules():
        print(rule)
    print("")

    port = int(os.environ.get("PORT", 8080))
    server.run(host="0.0.0.0", port=port)
