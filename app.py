from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import io

app = FastAPI()

ENGINE_NAME = "hsv_v29_geometry_left_span_locked"
API_KEY = "ithrive_secure_2026_key"

ROWS_PER_PAGE = 14

# Row indexes (0-based)
PAGE1_HEADING_ROWS = {0, 8}
PAGE2_HEADING_ROWS = {0, 8}

PAGE1_DISEASE_KEYS = [
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
]

PAGE2_DISEASE_KEYS = [
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

def map_percent_to_label(percent):
    if percent <= 10:
        return "normal"
    elif percent <= 20:
        return "none"
    elif percent <= 50:
        return "mild"
    elif percent <= 75:
        return "moderate"
    else:
        return "severe"

def detect_overlay_span(row_img):
    """
    Detect colored overlay width inside a single disease row.
    Overlay is left-aligned.
    """

    height, width, _ = row_img.shape

    # Only scan middle vertical band (ignore top/bottom padding)
    y1 = int(height * 0.25)
    y2 = int(height * 0.75)
    scan = row_img[y1:y2, :]

    # Convert to HSV
    hsv = cv2.cvtColor(scan, cv2.COLOR_BGR2HSV)

    # Define color ranges (striped colors)
    masks = []

    # Yellow
    masks.append(cv2.inRange(hsv, (20, 80, 80), (35, 255, 255)))

    # Orange
    masks.append(cv2.inRange(hsv, (10, 100, 100), (20, 255, 255)))

    # Red
    masks.append(cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)))

    # Neutral light gray (striped)
    masks.append(cv2.inRange(hsv, (0, 0, 160), (180, 40, 255)))

    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    # Sum mask vertically to find span
    column_sum = np.sum(combined, axis=0)

    # Threshold to determine if column contains overlay
    threshold = np.max(column_sum) * 0.3
    overlay_columns = np.where(column_sum > threshold)[0]

    if len(overlay_columns) == 0:
        return 20  # fallback baseline

    left = overlay_columns[0]
    right = overlay_columns[-1]

    span_ratio = (right - left) / width
    percent = int(round(span_ratio * 100 / 10) * 10)

    if percent < 10:
        percent = 10
    if percent > 100:
        percent = 100

    return percent

@app.post("/v1/detect-disease-bars")
async def detect_disease_bars(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    pdf_bytes = await file.read()
    pages = convert_from_bytes(pdf_bytes, dpi=200)

    results = {}

    # We only process first 2 relevant pages
    relevant_pages = pages[:2]

    for page_index, page in enumerate(relevant_pages):

        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        height, width, _ = img.shape
        row_height = height // ROWS_PER_PAGE

        disease_row_counter = 0

        for row_index in range(ROWS_PER_PAGE):

            y1 = row_index * row_height
            y2 = (row_index + 1) * row_height
            row_img = img[y1:y2, :]

            # Skip headings
            if page_index == 0 and row_index in PAGE1_HEADING_ROWS:
                continue
            if page_index == 1 and row_index in PAGE2_HEADING_ROWS:
                continue

            if page_index == 0:
                disease_key = PAGE1_DISEASE_KEYS[disease_row_counter]
            else:
                disease_key = PAGE2_DISEASE_KEYS[disease_row_counter]

            percent = detect_overlay_span(row_img)

            results[disease_key] = {
                "progression_percent": percent,
                "risk_label": map_percent_to_label(percent),
                "source": ENGINE_NAME
            }

            disease_row_counter += 1

    return JSONResponse({
        "engine": ENGINE_NAME,
        "pages_found": 2,
        "results": results
    })
