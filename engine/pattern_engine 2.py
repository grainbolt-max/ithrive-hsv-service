PATTERN_RULES = {
    "vascular_stress": [
        "large_artery_stiffness",
        "peripheral_vessel",
        "blood_pressure_uncontrolled",
        "atherosclerosis",
        "ldl_cholesterol"
    ],
    "metabolic_stress": [
        "metabolic_syndrome",
        "insulin_resistance",
        "beta_cell_function_decreased",
        "blood_glucose_uncontrolled"
    ],
    "inflammatory_pattern": [
        "tissue_inflammatory_process",
        "respiratory_disorders",
        "chronic_hepatitis"
    ],
    "hormonal_pattern": [
        "hypothyroidism",
        "hyperthyroidism"
    ],
    "neurological_pattern": [
        "major_depression",
        "cerebral_dopamine_decreased",
        "cerebral_serotonin_decreased",
        "adhd_children_learning"
    ]
}

COLOR_WEIGHT = {
    "grey": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3
}

def detect_patterns(disease_scores):

    patterns = {}

    for pattern, diseases in PATTERN_RULES.items():

        score = 0

        for disease in diseases:
            color = disease_scores.get(disease, "grey")
            score += COLOR_WEIGHT.get(color, 0)

        if score >= 4:
            patterns[pattern] = "active"
        elif score >= 2:
            patterns[pattern] = "mild"
        else:
            patterns[pattern] = "inactive"

    return patterns