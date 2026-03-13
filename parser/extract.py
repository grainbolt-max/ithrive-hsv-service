import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v71_visible_proof"

X_LEFT = 937
MIN_Y = 880
MAX_Y = 2050

DISEASES = [
"large_artery_stiffness","peripheral_vessel","blood_pressure_uncontrolled",
"small_medium_artery_stiffness","atherosclerosis","ldl_cholesterol",
"lv_hypertrophy","metabolic_syndrome","insulin_resistance",
"beta_cell_function_decreased","blood_glucose_uncontrolled",
"tissue_inflammatory_process","hypothyroidism","hyperthyroidism",
"hepatic_fibrosis","chronic_hepatitis","prostate_cancer",
"respiratory_disorders","kidney_function_disorders","digestive_disorders",
"major_depression","adhd_children_learning","cerebral_dopamine_decreased",
"cerebral_serotonin_decreased"
]

def detect_rows(img):
    DETECT_X = X_LEFT - 12
    column = img[MIN_Y:MAX_Y, DETECT_X:DETECT_X + 45]
    gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)
    rows = []
    inside = False
    start = 0
    for y in range(gray.shape[0]):
        intensity = np.mean(gray[y, :])
        if intensity < 225 and not inside:
            start = y
            inside = True
        if intensity > 240 and inside:
            end = y
            if 10 < (end - start) < 40:
                rows.append((start + MIN_Y, end + MIN_Y))
            inside = False
    filtered = []
    for r in rows:
        if not filtered or r[0] - filtered[-1][0] > 18:
            filtered.append(r)
    return filtered

def sample_bar_color(img, y1, y2):
    mid = int((y1 + y2) / 2)
    bar_zone = img[mid-4:mid+4, X_LEFT-90 : X_LEFT+60]
    hsv = cv2.cvtColor(bar_zone, cv2.COLOR_BGR2HSV)
    for x in range(hsv.shape[1]):
        if hsv[0, x, 2] < 245:   # any non-white pixel
            return np.array([int(hsv[0,x,0]), int(hsv[0,x,1]), int(hsv[0,x,2])])
    return np.array([0, 0, 255])

def classify_bar(sample):
    h, s, v = sample
    if s < 30:
        return "grey"
    if h < 12 or h > 168:
        return "red"
    elif 12 <= h < 37:
        return "orange"
    elif 37 <= h <= 88:
        return "yellow"
    return "grey"

def draw_debug(img, rows, scores):
    debug = img.copy()
    # PROOF TEXT - you WILL see this
    cv2.putText(debug, f"ENGINE v71 - ROWS DETECTED: {len(rows)}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
    
    colors = {"grey":(128,128,128), "yellow":(0,255,255),
              "orange":(0,165,255), "red":(0,0,255)}
    
    for i, (y1, y2) in enumerate(rows):
        if i >= len(DISEASES):
            break
        risk = scores.get(DISEASES[i], "grey")
        # Rectangle on the bar
        cv2.rectangle(debug, (X_LEFT, y1), (X_LEFT + 22, y2), colors[risk], 5)
        # Label next to bar
        cv2.putText(debug, risk.upper(), (X_LEFT + 40, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, colors[risk], 4)
    
    return debug

def parse_report(pdf_bytes, debug=False):
    images = convert_from_bytes(pdf_bytes, dpi=200)
    img = np.array(images[1])   # diseases page
    
    rows = detect_rows(img)
    samples = [sample_bar_color(img, y1, y2) for y1, y2 in rows]
    
    scores = {}
    for i, (y1, y2) in enumerate(rows):
        if i >= len(DISEASES):
            break
        scores[DISEASES[i]] = classify_bar(samples[i])
    
    if debug:
        overlay = draw_debug(img, rows, scores)
        _, png = cv2.imencode(".png", overlay)
        return png.tobytes()
    
    return {"engine": ENGINE_NAME, "scores": scores}

def extract_scores(pdf_bytes):
    return parse_report(pdf_bytes)
