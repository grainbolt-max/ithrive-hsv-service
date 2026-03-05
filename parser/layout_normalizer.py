import numpy as np


def normalize_layout(img, anchors):

    h, w, _ = img.shape

    layout = {}

    # convert anchors to relative coordinates
    layout["header_y"] = anchors["header_y"] / h
    layout["footer_y"] = anchors["footer_y"] / h
    layout["left_x"] = anchors["left_x"] / w
    layout["risk_bar_x"] = anchors["risk_bar_x"] / w

    return layout


def resolve_layout(img, layout):

    h, w, _ = img.shape

    anchors = {}

    anchors["header_y"] = int(layout["header_y"] * h)
    anchors["footer_y"] = int(layout["footer_y"] * h)
    anchors["left_x"] = int(layout["left_x"] * w)
    anchors["risk_bar_x"] = int(layout["risk_bar_x"] * w)

    return anchors