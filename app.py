from flask import Flask, request, jsonify, Response, send_from_directory
import os

from parser.extract import extract_scores

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY", "ithrive_secure_2026_key")


@app.route("/")
def root():
    return {"status": "ok", "service": "ithrive-hsv-service"}


@app.route("/docs")
def docs():
    return send_from_directory("static", "docs.html")


@app.route("/debug-crop")
def debug_crop():
    return send_from_directory("/tmp", "debug_crop.png")


@app.route("/parse-report", methods=["POST"])
def parse_report():

    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    debug = request.form.get("debug") in ["true", "1", "yes"]

    try:
        result = extract_scores(pdf_bytes, debug=debug)
    except Exception as e:
        return jsonify({
            "error": "parser_failure",
            "message": str(e)
        }), 500

    if debug:
        return Response(result, mimetype="image/png")

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
