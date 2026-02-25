from flask import Flask, request, jsonify
import fitz  # PyMuPDF

app = Flask(__name__)

ENGINE_NAME = "v100_vector_rectangle_production_classifier"
API_KEY = "ithrive_secure_2026_key"
PAGE_INDEX = 1  # Page 2
EXPECTED_ROWS = 24

DISEASES = [
    "adhd_children_learning",
    "atherosclerosis",
    "beta_cell_function_decreased",
    "blood_glucose_uncontrolled",
    "blood_pressure_uncontrolled",
    "cerebral_dopamine_serotonin",
    "chronic_hepatitis",
    "diabetes",
    "digestive_disorders",
    "hepatic_fibrosis",
    "hyperthyroidism",
    "hypothyroidism",
    "insulin_resistance",
    "kidney_function_disorders",
    "large_artery_stiffness",
    "ldl_cholesterol",
    "lv_hypertrophy",
    "major_depression",
    "metabolic_syndrome",
    "peripheral_vessel",
    "prostate_cancer",
    "respiratory_disorders",
    "small_medium_artery_stiffness",
    "tissue_inflammatory_process"
]


def classify_rgb(r, g, b):
    # Convert 0-1 floats to 0-255 scale if needed
    if r <= 1 and g <= 1 and b <= 1:
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

    # Light grey / none
    if abs(r - g) < 10 and abs(g - b) < 10:
        return "none"

    # Severe (red)
    if r > 180 and g < 100:
        return "severe"

    # Moderate (orange)
    if r > 180 and 100 <= g <= 170:
        return "moderate"

    # Mild (yellow)
    if r > 180 and g > 170:
        return "mild"

    return "none"


@app.route("/")
def home():
    return f"HSV Preprocess Service Running {ENGINE_NAME}"


@app.route("/v1/detect-disease-bars", methods=["POST"])
def detect():

    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except:
        return jsonify({"error": "Invalid PDF"}), 400

    if len(doc) <= PAGE_INDEX:
        return jsonify({"error": "Page 2 not found"}), 400

    page = doc[PAGE_INDEX]
    drawings = page.get_drawings()

    bars = []

    for d in drawings:
        if d["type"] != "f":
            continue  # only filled shapes

        fill = d.get("fill")
        rect = d.get("rect")

        if not fill or not rect:
            continue

        width = rect.width
        height = rect.height

        # Filter to horizontal stripe bars only
        if width < 200:      # too narrow
            continue
        if height < 8 or height > 40:  # unrealistic row height
            continue

        # Ignore black / borders
        if fill == (0, 0, 0):
            continue

        bars.append({
            "y": rect.y0,
            "fill": fill,
            "rect": rect
        })

    # Remove duplicates by Y proximity
    bars = sorted(bars, key=lambda x: x["y"])
    unique_bars = []

    for b in bars:
        if not unique_bars:
            unique_bars.append(b)
            continue

        if abs(b["y"] - unique_bars[-1]["y"]) > 5:
            unique_bars.append(b)

    if len(unique_bars) < EXPECTED_ROWS:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": f"Detected {len(unique_bars)} bars, expected {EXPECTED_ROWS}"
        })

    unique_bars = unique_bars[:EXPECTED_ROWS]

    results = {}

    for i in range(EXPECTED_ROWS):
        fill = unique_bars[i]["fill"]
        r, g, b = fill

        severity = classify_rgb(r, g, b)

        results[DISEASES[i]] = {
            "risk_label": severity,
            "source": ENGINE_NAME
        }

    return jsonify({
        "engine": ENGINE_NAME,
        "page_index_processed": PAGE_INDEX,
        "bars_detected": len(unique_bars),
        "results": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
