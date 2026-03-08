import hashlib
import cv2


def compute_layout_hash(page):

    small = cv2.resize(page, (200, 200))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    pixel_bytes = gray.tobytes()

    layout_hash = hashlib.sha256(pixel_bytes).hexdigest()

    return layout_hash


def choose_parser(page):

    layout_hash = compute_layout_hash(page)

    # known layout (your current working report)
    KNOWN_LAYOUT = "YlLY455AeeVlZXU8xGy1yd04QIomu+5OyCOaFw+8oHg="

    if layout_hash.startswith(KNOWN_LAYOUT[:10]):
        return "HSV"

    return "SEMANTIC"