SYSTEM_MAP = {

"Metabolic Regulation":[
"Metabolic syndrome",
"Insulin resistance",
"Blood glucose uncontrolled",
"Impaired glucose tolerance",
"Glucose transport dysfunction",
"Fatty acid metabolism disorder"
],

"Inflammatory Control":[
"Chronic inflammatory state",
"Systemic inflammatory activity",
"Inflammatory cytokine elevation",
"Tissue inflammatory process",
"Immune imbalance"
],

"Vascular Function":[
"Endothelial dysfunction",
"Microvascular circulation disorder",
"Peripheral circulation impairment",
"Large artery stiffness",
"Vascular elasticity reduction"
],

"Hormonal Endocrine":[
"Endocrine metabolic dysregulation",
"Hormonal metabolic imbalance",
"Pancreatic endocrine disorder",
"Insulin signaling impairment"
],

"Cellular Energy":[
"Cellular metabolic stress",
"Cellular energy metabolism issue",
"Chronic oxidative pressure",
"Oxidative stress elevation"
]

}


def calculate_system_scores(disease_scores):

    systems = {}

    for system, markers in SYSTEM_MAP.items():

        values = []

        for m in markers:
            values.append(disease_scores.get(m,0))

        if len(values) == 0:
            score = 0
        else:
            score = sum(values)/len(values)

        systems[system] = round(score,3)

    return systems


def detect_root_drivers(system_scores):

    drivers = []

    if system_scores.get("Metabolic Regulation",0) > 0.30:
        drivers.append("Glucose regulation stress")

    if system_scores.get("Inflammatory Control",0) > 0.30:
        drivers.append("Chronic inflammatory signaling")

    if system_scores.get("Vascular Function",0) > 0.30:
        drivers.append("Vascular endothelial strain")

    if system_scores.get("Cellular Energy",0) > 0.25:
        drivers.append("Mitochondrial energy stress")

    return drivers


def rank_intervention_priorities(system_scores):

    ranked = sorted(
        system_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    priorities = []

    for r in ranked[:3]:
        priorities.append(r[0])

    return priorities


def interpret_scan(disease_scores):

    systems = calculate_system_scores(disease_scores)

    drivers = detect_root_drivers(systems)

    priorities = rank_intervention_priorities(systems)

    return {
        "systems": systems,
        "root_drivers": drivers,
        "priorities": priorities
    }