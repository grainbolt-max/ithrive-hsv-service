from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from pdf2image import convert_from_bytes
import os

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

    import cv2
    import numpy as np
    from flask import Response

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

    # draw anchors
    for name, (x, y) in anchors.items():
        cv2.circle(debug_img, (int(x), int(y)), 10, (255,0,0), -1)

    # draw row boxes
    for r in rows:
        x1, y1, x2, y2 = r
        cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0,255,0), 2)

    _, buffer = cv2.imencode(".png", debug_img)

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
    narrative = generate_health_narrative(system_summary, consultation_summary, protocol)

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
