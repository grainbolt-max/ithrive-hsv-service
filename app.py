from flask import Flask, request, jsonify
import fitz  # PyMuPDF

app = Flask(__name__)

ENGINE_NAME = "vStructure_Inspector"
API_KEY = "ithrive_secure_2026_key"
PAGE_INDEX = 1  # Page 2 (0-based index)


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def inspect_pdf_structure():

    # --- Auth ---
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    # --- File Check ---
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()

    # --- Open PDF ---
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return jsonify({"error": f"Invalid PDF: {str(e)}"}), 400

    if len(doc) <= PAGE_INDEX:
        return jsonify({"error": "Page 2 not found"}), 400

    page = doc[PAGE_INDEX]

    # --- Extract Structural Elements ---
    drawings = page.get_drawings()
    images = page.get_images(full=True)
    text_blocks = page.get_text("blocks")
    xobjects = page.get_xobjects()

    # --- Basic Geometry Info ---
    page_rect = page.rect

    return jsonify({
        "engine": ENGINE_NAME,
        "page_index": PAGE_INDEX,
        "page_width": page_rect.width,
        "page_height": page_rect.height,
        "drawings_count": len(drawings),
        "images_count": len(images),
        "xobjects_count": len(xobjects),
        "text_blocks_count": len(text_blocks),
        "sample_drawing_types": list(set([d.get("type") for d in drawings]))[:10],
        "sample_image_refs": [img[0] for img in images[:5]]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
