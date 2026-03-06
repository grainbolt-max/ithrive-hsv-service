import json
import hashlib
import os

REGISTRY_FILE = "parser/layout_registry.json"


def fingerprint_layout(image, anchors, rows):

    data = {
        "width": image.shape[1],
        "height": image.shape[0],
        "anchors": anchors,
        "row_count": len(rows)
    }

    raw = json.dumps(data, sort_keys=True).encode()

    return hashlib.sha1(raw).hexdigest()


def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return {}

    try:
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=int)


def register_layout(layout_hash, anchors, rows):

    registry = load_registry()

    if layout_hash not in registry:

        registry[layout_hash] = {
            "anchors": anchors,
            "rows": rows
        }

        save_registry(registry)

    return registry[layout_hash]