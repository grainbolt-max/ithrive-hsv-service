from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import json

app = Flask(__name__)

ENGINE_NAME = "v82_pdf_diagnostic_extractor"
API_KEY = "ithrive_secure_2026_key"

# ==================================================
# DIAGNOSTIC EXTRACTION
# ==================================================

def extract_page_diagnostics(page):

    drawings = page.get_drawings()
    xobjects = page.get_xobjects()
    text_raw = page.get_text("rawdict")

    drawing_summary = []

    for d in drawings:
        entry = {
            "type": d.get("type"),
            "stroke_color": d.get("color"),
            "fill_color": d.get("fill"),
            "width": d.get("width"),
            "items_count": len(d.get("items", [])),
            "rect": str(d.get("rect"))
        }
        drawing_summary.append(entry)

    xobject_summary = []

    for xo in xobjects:
        xobject_summary.append({
            "xref": xo[0],
            "name": xo[1],
            "width": xo[2],
            "height": xo[3]
        })

    return {
        "drawings_count": len(drawings),
        "drawings": drawing_summary[:100],  # limit size
        "xobjects_count": len(xobjects),
        "xobjects": xobject_summary,
        "text_blocks": len(text_raw.get("blocks", []))
    }


# ==================================================
# MAIN
# ==================================================

def diagnose_pdf(pdf_bytes):

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    results = {}

    for page_index in [0, 1]:
        page = doc[page_index]
        results[f"page_{page_index+1}"] = extract_page_diagnostics(page)

    return results


# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect_disease_bars():

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    try:
        diagnostics = diagnose_pdf(pdf_bytes)

        return jsonify({
            "engine": ENGINE_NAME,
            "diagnostics": diagnostics
        })

    except Exception as e:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
