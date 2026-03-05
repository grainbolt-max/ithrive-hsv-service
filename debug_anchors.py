import cv2
import numpy as np
from pdf2image import convert_from_path
from parser.anchors import detect_all_anchors
from parser.rows import detect_row_lines
from parser.extract import extract_disease_scores

PDF_FILE = "sample.pdf"

def draw_anchor(img, x, y, label):
    cv2.circle(img, (x, y), 12, (0, 0, 255), -1)
    cv2.putText(img, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():

    pages = convert_from_path(PDF_FILE)

    page = np.array(pages[1])

    img = cv2.cvtColor(page, cv2.COLOR_RGB2BGR)

    anchors = detect_all_anchors(img)

    rows = detect_row_lines(
        img,
        anchors["header_y"],
        anchors["table_bottom_y"]
    )

    print("Detected row lines:", rows)

    scores = extract_disease_scores(img, rows, anchors)

    print("Extracted disease scores:")
    print(scores)

    h, w, _ = img.shape

    for y in rows:
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)

    draw_anchor(img, 50, anchors["header_y"], "HEADER")
    draw_anchor(img, anchors["left_x"], 200, "LEFT COLUMN")
    draw_anchor(img, anchors["risk_bar_x"], 400, "RISK BAR")
    draw_anchor(img, 200, anchors["table_bottom_y"], "TABLE BOTTOM")

    if anchors["footer_y"] is not None:
        draw_anchor(img, 200, anchors["footer_y"], "FOOTER")

    cv2.imwrite("anchor_debug.png", img)

    print("Anchors detected:")
    print(anchors)

    print("Debug image saved: anchor_debug.png")

if __name__ == "__main__":
    main()
