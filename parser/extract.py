from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import numpy as np
from pdf2image import convert_from_bytes
import os
import cv2

from parser.layout_normalizer import normalize_dpi
from parser.system_engine import compute_system_summary, compute_consultation_summary

from engine.pattern_engine import detect_patterns
from engine.protocol_engine import build_protocol
from engine.narrative_engine import generate_health_narrative

server = Flask(__name__, static_folder="static")
CORS(server)

ENGINE_NAME = "ithrive_disease_parser_v1"
API_KEY = "ithrive_secure_2026_key"

# =====================================================
# FIXED SAMPLING WINDOW (visualized in debug endpoint)
# =====================================================

SCORE_SAMPLE_LEFT = 904
SCORE_SAMPLE_RIGHT = 906

# =====================================================
# ROOT
# =====================================================

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


# =====================================================
# DEBUG VISUALIZER
# =====================================================

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

    debug_img = page.copy()

    height, width = debug_img.shape[:2]

    # =====================================
    # DRAW X GRID EVERY 25px
    # =====================================

    for x in range(0, width, 25):

        cv2.line(
            debug_img,
            (x, 0),
            (x, height),
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

    # =====================================
    # DRAW SAMPLING STRIPE
    # =====================================

    cv2.rectangle(
        debug_img,
        (SCORE_SAMPLE_LEFT, 0),
        (SCORE_SAMPLE_RIGHT, height),
        (255, 0, 255),
        3
    )

    success, buffer = cv2.imencode(".png", debug_img)

    if not success:
        return "image encoding failed", 500

    return Response(buffer.tobytes(), mimetype="image/png")


# =====================================================
# PARSE REPORT
# =====================================================

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

    # =================================================
    # DISEASE EXTRACTION (DETERMINISTIC)
    # =================================================

    scores = extract_disease_scores(page)

    # =================================================
    # PATTERN DETECTION
    # =================================================

    patterns = detect_patterns(scores)

    # =================================================
    # PROTOCOL BUILDER
    # =================================================

    protocol = build_protocol(patterns)

    # =================================================
    # SYSTEM SUMMARIES
    # =================================================

    system_summary = compute_system_summary(scores)
    consultation_summary = compute_consultation_summary(system_summary)

    # =================================================
    # NARRATIVE ENGINE
    # =================================================

    narrative = generate_health_narrative(
        system_summary,
        consultation_summary,
        protocol
    )

    return jsonify({
        "engine": ENGINE_NAME,
        "system_summary": system_summary,
        "consultation_summary": consultation_summary,
        "protocol": protocol,
        "health_narrative": narrative,
        "disease_scores": scores
    })


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":

    print("\nREGISTERED ROUTES:")
    for rule in server.url_map.iter_rules():
        print(rule)

    print("")

    port = int(os.environ.get("PORT", 8080))

    server.run(
        host="0.0.0.0",
        port=port
    )
