PROTOCOLS = {

"cardiovascular":{

"exercise":{
"cardio_priority":True,
"cardio_minutes":30,
"strength_sessions":2,
"mobility_focus":True
},

"nutrition":{
"focus_foods":["salmon","olive oil","leafy greens","berries","avocado"],
"avoid":["processed_meats","fried_foods","excess_sodium"],
"macro_style":"balanced"
}

},

"metabolic":{

"exercise":{
"strength_priority":True,
"strength_sessions":4,
"cardio_minutes":20
},

"nutrition":{
"focus_foods":["lean_protein","vegetables","whole_grains"],
"avoid":["refined_sugar","sweetened_drinks"],
"macro_style":"high_protein"
}

},

"inflammatory":{

"exercise":{
"intensity":"low",
"mobility_focus":True
},

"nutrition":{
"focus_foods":["berries","turmeric","fatty_fish","olive_oil"],
"avoid":["processed_foods","trans_fats"],
"macro_style":"anti_inflammatory"
}

},

"hormonal":{

"exercise":{
"strength_priority":True,
"recovery_days":True
},

"nutrition":{
"focus_foods":["healthy_fats","fiber","omega3"],
"avoid":["refined_carbs"],
"macro_style":"balanced"
}

},

"neurological":{

"exercise":{
"balance_training":True,
"mobility_focus":True
},

"nutrition":{
"focus_foods":["omega3","nuts","dark_chocolate","leafy_greens"],
"avoid":["high_sugar"],
"macro_style":"brain_support"
}

}

}

def build_protocol(patterns):

    protocol={
        "exercise_rules":{},
        "nutrition_rules":{}
    }

    for pattern,status in patterns.items():

        if status!="active":
            continue

        if pattern=="vascular_stress":
            system="cardiovascular"

        elif pattern=="metabolic_stress":
            system="metabolic"

        elif pattern=="inflammatory_pattern":
            system="inflammatory"

        elif pattern=="hormonal_pattern":
            system="hormonal"

        elif pattern=="neurological_pattern":
            system="neurological"

        else:
            continue

        system_protocol=PROTOCOLS.get(system)

        if not system_protocol:
            continue

        protocol["exercise_rules"].update(system_protocol["exercise"])
        protocol["nutrition_rules"].update(system_protocol["nutrition"])

    return protocol