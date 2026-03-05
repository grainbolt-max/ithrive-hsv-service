import json
import os

REGISTRY_FILE = "layout_registry.json"


def load_registry():

    if not os.path.exists(REGISTRY_FILE):
        return {}

    with open(REGISTRY_FILE,"r") as f:
        return json.load(f)


def save_layout(layout_name,fingerprint):

    registry = load_registry()

    registry[layout_name] = fingerprint

    with open(REGISTRY_FILE,"w") as f:
        json.dump(registry,f,indent=2)