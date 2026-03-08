from parser.disease_list import DISEASE_LIST

def validate_parser_output(data):
    if not isinstance(data, dict):
        raise ValueError("Parser output must be a dictionary")

    expected = set(DISEASE_LIST)
    returned = set(data.keys())

    missing = expected - returned
    extra = returned - expected

    if missing:
        raise ValueError(f"Parser missing disease keys: {missing}")

    if extra:
        raise ValueError(f"Unexpected disease keys returned: {extra}")
