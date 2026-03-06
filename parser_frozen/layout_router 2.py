import numpy as np


def compute_layout_fingerprint(img, anchors):

    h, w, _ = img.shape

    fingerprint = {
        "aspect_ratio": round(w / h, 3),
        "header_pos": round(anchors["header_y"] / h, 3),
        "footer_pos": round(anchors["footer_y"] / h, 3),
        "left_col": round(anchors["left_x"] / w, 3),
        "risk_bar_col": round(anchors["risk_bar_x"] / w, 3)
    }

    return fingerprint


def identify_layout(fingerprint):

    known_layouts = {
        "bio_scan_v1": {
            "aspect_ratio": 0.773,
            "header_pos": 0.026,
            "left_col": 0.075,
            "risk_bar_col": 0.153
        }
    }

    for layout_name, ref in known_layouts.items():

        score = 0

        if abs(fingerprint["aspect_ratio"] - ref["aspect_ratio"]) < 0.05:
            score += 1

        if abs(fingerprint["header_pos"] - ref["header_pos"]) < 0.05:
            score += 1

        if abs(fingerprint["left_col"] - ref["left_col"]) < 0.05:
            score += 1

        if abs(fingerprint["risk_bar_col"] - ref["risk_bar_col"]) < 0.05:
            score += 1

        if score >= 3:
            return layout_name

    return "unknown_layout"