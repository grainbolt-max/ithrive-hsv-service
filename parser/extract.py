import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v70_header_anchor_classifier"

# column where the risk bars exist
X_LEFT = 939
X_RIGHT = 954

# diseases in order
DISEASES = [
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
    "tissue_inflammatory_process",
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
    "cerebral_serotonin_decreased",
]


# -------------------------------------------------
# Detect disease table rows using header anchor
# -------------------------------------------------

def detect_rows(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # search area where disease header exists
    header_region = gray[900:1100, 400:1600]

    edges = cv2.Canny(header_region, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=600,
        maxLineGap=10
    )

    header_y = None

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]

            if abs(y1-y2) < 3 and (x2-x1) > 800:
                header_y = y1 + 900
                break

    if header_y is None:
        raise Exception("Could not detect disease table header")

    # offsets from the header to each row center
    row_offsets = [
        120,165,210,255,300,
        345,390,450,495,540,
        585,630,700,745,790,
        835,880,925,970,1015,
        1060,1105,1150,1195
    ]

    rows = [header_y + offset for offset in row_offsets]

    return rows


# -------------------------------------------------
# Sample color from risk bar
# -------------------------------------------------

def sample_bar_color(img, y):

    crop = img[y-6:y+6, X_LEFT:X_RIGHT]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = np.mean(hsv[:,:,0])
    s = np.mean(hsv[:,:,1])
    v = np.mean(hsv[:,:,2])

    return h,s,v


# -------------------------------------------------
# Classify HSV color
# -------------------------------------------------

def classify_bar(h,s,v):

    if s < 40:
        return None

    if h < 10:
        return "red"

    if h < 25:
        return "orange"

    if h < 45:
        return "yellow"

    return None


# -------------------------------------------------
# Debug overlay
# -------------------------------------------------

def draw_debug(img, rows):

    overlay = img.copy()

    for y in rows:

        cv2.rectangle(
            overlay,
            (X_LEFT-10, y-8),
            (X_RIGHT+10, y+8),
            (0,255,0),
            2
        )

    return overlay


# -------------------------------------------------
# Main parser
# -------------------------------------------------

def parse_report(pdf_bytes, debug=False):

    pages = convert_from_bytes(pdf_bytes, dpi=200)

    if len(pages) < 2:
        raise Exception("PDF missing disease screening page")

    page = pages[1]

    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    rows = detect_rows(img)

    scores = {}

    for disease, y in zip(DISEASES, rows):

        color = sample_bar_color(img, y)

        if color is None:
            scores[disease] = None
            continue

        h,s,v = color

        scores[disease] = classify_bar(h,s,v)

    if debug:

        overlay = draw_debug(img, rows)

        ok, png = cv2.imencode(".png", overlay)

        return png.tobytes()

    ordered_scores = {d: scores.get(d) for d in DISEASES}

    return {
        "engine": ENGINE_NAME,
        "scores": ordered_scores
    }


# -------------------------------------------------
# External entrypoint
# -------------------------------------------------

def extract_scores(pdf_bytes, debug=False):

    return parse_report(pdf_bytes, debug=debug)
