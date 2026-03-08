SYSTEM_MAP = {
    "cardiovascular":[
        "large_artery_stiffness",
        "small_medium_artery_stiffness",
        "peripheral_vessel",
        "blood_pressure_uncontrolled",
        "atherosclerosis",
        "ldl_cholesterol",
        "lv_hypertrophy"
    ],

    "metabolic":[
        "metabolic_syndrome",
        "insulin_resistance",
        "beta_cell_function_decreased",
        "blood_glucose_uncontrolled"
    ],

    "inflammatory":[
        "tissue_inflammatory_process",
        "digestive_disorders",
        "respiratory_disorders",
        "chronic_hepatitis",
        "hepatic_fibrosis"
    ],

    "hormonal":[
        "hypothyroidism",
        "hyperthyroidism",
        "prostate_cancer",
        "kidney_function_disorders"
    ],

    "neurological":[
        "major_depression",
        "adhd_children_learning",
        "cerebral_dopamine_decreased",
        "cerebral_serotonin_decreased"
    ]
}

COLOR_SCORE = {
    "grey":0,
    "yellow":1,
    "orange":2,
    "red":3
}

SCORE_LABEL = {
    0:"low",
    1:"mild",
    2:"moderate",
    3:"severe"
}

def compute_system_summary(disease_scores):

    system_summary = {}

    for system, diseases in SYSTEM_MAP.items():

        max_score = 0

        for d in diseases:
            if d in disease_scores:

                score = COLOR_SCORE.get(disease_scores[d],0)

                if score > max_score:
                    max_score = score

        system_summary[system] = SCORE_LABEL[max_score]

    return system_summary


def compute_consultation_summary(system_summary):

    score_rank = {
        "low":0,
        "mild":1,
        "moderate":2,
        "severe":3
    }

    ranked = sorted(
        system_summary.items(),
        key=lambda x: score_rank[x[1]],
        reverse=True
    )

    primary = ranked[0][0]
    secondary = ranked[1][0]

    return {
        "primary_driver":primary,
        "secondary_driver":secondary
    }