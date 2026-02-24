from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import numpy as np

app = Flask(__name__)

# ==================================================
# DECLARE
# ==================================================

ENGINE_NAME = "v80_full_vector_stroke_engine"
API_KEY = "ithrive_secure_2026_key"

PAGE_1_DISEASES = [
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
    "tissue_inflammatory_process"
]

PAGE_2_DISEASES = [
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
    "cerebral_serotonin_decreased"
]

# ==================================================
# COLOR CLASSIFICATION (RGB)
# ==================================================

def classify_rgb(rgb):
    if rgb is None:
        return "none"

    r, g, b = rgb

    # Grey detection (channels nearly equal)
    if abs(r - g) < 0.05 and abs(g - b) < 0.05:
        return "none"

    # Severe (Red dominant)
    if r > 0.7 and g < 0.4:
        return "severe"

    # Moderate (Orange)
    if r > 0.7 and 0.4 <= g <= 0.7:
        return "moderate"

    # Mild (Yellow)
    if r > 0.7 and g > 0.7:
        return "mild"

    return "none"


# ==================================================
# VECTOR EXTRACTION
# ==================================================

def extract_bar_stroke_colors(page):

    drawings = page.get_drawings()

    vertical_lines = []

    for d in drawings:
        if d["type"] != "stroke":
            continue

        color = d.get("color")
        items = d.get("items", [])

        for item in items:
            if item[0] == "l":  # line
                x1, y1, x2, y2 = item[1]

                # vertical line detection
                if abs(x1 - x2) < 1 and abs(y2 - y1) > 10:
                    vertical_lines.append({
                        "y_mid": (y1 + y2) / 2,
                        "color": color
                    })

    if not vertical_lines:
        return []

    # Group by Y proximity into bars
    vertical_lines.sort(key=lambda x: x["y_mid"])

    bars = []
    current_group = [vertical_lines[0]]

    for line in vertical_lines[1:]:
        if abs(line["y_mid"] - current_group[-1]["y_mid"]) < 15:
            current_group.append(line)
        else:
            bars.append(current_group)
            current_group = [line]

    bars.append(current_group)

    # For each bar, determine dominant color
    bar_colors = []

    for group in bars:
        colors = [tuple(line["color"]) for line in group if line["color"]]

        if not colors:
            bar_colors.append(None)
            continue

        # Most frequent color in this group
        unique, counts = np.unique(colors, axis=0, return_counts=True)
        dominant = unique[np.argmax(counts)]

        bar_colors.append(tuple(dominant))

    return bar_colors


# ==================================================
# MAIN EXTRACTION
# ==================================================

def detect_all_24_diseases(pdf_bytes):

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page1 = doc[0]
    page2 = doc[1]

    bars_page1 = extract_bar_stroke_colors(page1)
    bars_page2 = extract_bar_stroke_colors(page2)

    if len(bars_page1) < 12 or len(bars_page2) < 12:
        raise RuntimeError("Did not detect 12 stroke groups per page")

    results = {}

    # Page 1 direct order
    for disease, color in zip(PAGE_1_DISEASES, bars_page1[:12]):
        results[disease] = {
            "risk_label": classify_rgb(color),
            "source": ENGINE_NAME
        }

    # Page 2 reversed
    reversed_page2 = list(reversed(bars_page2[:12]))

    for disease, color in zip(PAGE_2_DISEASES, reversed_page2):
        results[disease] = {
            "risk_label": classify_rgb(color),
            "source": ENGINE_NAME
        }

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
        results = detect_all_24_diseases(pdf_bytes)

        return jsonify({
            "engine": ENGINE_NAME,
            "results": results
        })

    except Exception as e:
        return jsonify({
            "engine": ENGINE_NAME,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
