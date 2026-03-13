import cv2
import numpy as np
from pdf2image import convert_from_bytes

ENGINE_NAME = "v70_robust_bar_scan"

# BAR COLUMN (exact location on every ITHRIVE report at dpi=200)
X_LEFT = 937

# Detection window
MIN_Y = 880
MAX_Y = 2050

DISEASES = [ ... ]  # keep your exact list

# ------------------------------------------------
# ROW DETECTION (catches grey + colored bars)
# ------------------------------------------------
def detect_rows(img):
    DETECT_X = X_LEFT - 10          # 927 — slightly left of bar start
    column = img[MIN_Y:MAX_Y, DETECT_X:DETECT_X + 40]
    gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)
    
    rows = []
    inside = False
    start = 0
    for y in range(gray.shape[0]):
        intensity = np.mean(gray[y, :])
        if intensity < 225 and not inside:      # grey/colored bar
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


# ------------------------------------------------
# SAMPLE BAR COLOR (scans left-to-right until it hits ANY bar fill)
# ------------------------------------------------
def sample_bar_color(img, y1, y2):
    mid = int((y1 + y2) / 2)
    # Scan wide around the bar column (short yellow/grey bars start ~80px left)
    bar_zone = img[mid-3:mid+3, X_LEFT-80 : X_LEFT+50]
    hsv = cv2.cvtColor(bar_zone, cv2.COLOR_BGR2HSV)
    
    for x in range(hsv.shape[1]):
        if hsv[0, x, 2] < 245:          # not pure white background
            return np.array([int(hsv[0, x, 0]), int(hsv[0, x, 1]), int(hsv[0, x, 2])])
    return np.array([0, 0, 255])        # grey fallback


# ------------------------------------------------
# CLASSIFY (hardcoded + tolerant for light grey bars)
# ------------------------------------------------
def classify_bar(sample):
    h, s, v = sample
    if s < 30:                          # grey or empty
        return "grey"
    if h < 15 or h > 165:               # red
        return "red"
    elif 15 <= h < 38:                  # orange
        return "orange"
    elif 38 <= h <= 85:                 # yellow
        return "yellow"
    return "grey"


# ------------------------------------------------
# DEBUG DRAW (uses original X_LEFT so rectangles land exactly on the bars)
# ------------------------------------------------
def draw_debug(img, rows, scores):
    debug = img.copy()
    colors = {
        "grey":   (128, 128, 128),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "red":    (0, 0, 255)
    }
    for i, (y1, y2) in enumerate(rows):
        if i >= len(DISEASES):
            break
        risk = scores.get(DISEASES[i], "grey")
        cv2.rectangle(debug, (X_LEFT, y1), (X_LEFT + 18, y2), colors[risk], 4)  # thicker for visibility
    return debug


# ------------------------------------------------
# MAIN PARSER
# ------------------------------------------------
def parse_report(pdf_bytes, debug=False):
    images = convert_from_bytes(pdf_bytes, dpi=200)
    img = np.array(images[1])               # page 2 (0-based)
    
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
    
    return {
        "engine": ENGINE_NAME,
        "scores": scores
    }

def extract_scores(pdf_bytes):
    return parse_report(pdf_bytes)
