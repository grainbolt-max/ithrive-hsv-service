import re
import fitz
from flask import Flask, request, jsonify

API_KEY = "ithrive_secure_2026_key"
app = Flask(__name__)


def is_authorized(req):
    auth = req.headers.get("Authorization", "")
    return auth == f"Bearer {API_KEY}"


@app.route("/debug-text", methods=["POST"])
def debug_text():
    if not is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "file_missing"}), 400

    file = request.files["file"]
    pdf_bytes = file.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    doc.close()

    return jsonify({"raw_text": full_text})


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "debug_mode_running"})
