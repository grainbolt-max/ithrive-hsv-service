def generate_health_narrative(system_summary, consultation_summary, protocol):

    primary = consultation_summary.get("primary_driver", "")
    secondary = consultation_summary.get("secondary_driver", "")

    narrative = {}

    narrative["overview"] = (
        f"Your scan indicates that the primary health driver affecting your system "
        f"is related to {primary} stress patterns. "
        f"A secondary contributor appears to be {secondary} activity."
    )

    system_lines = []

    for system, level in system_summary.items():

        if level == "moderate":
            system_lines.append(
                f"The {system} system shows moderate stress and may benefit from targeted lifestyle improvements."
            )

        elif level == "mild":
            system_lines.append(
                f"The {system} system shows mild activity that should be monitored and supported through healthy habits."
            )

        else:
            system_lines.append(
                f"The {system} system appears stable at this time."
            )

    narrative["systems"] = system_lines

    exercise = protocol.get("exercise_rules", {})
    cardio_minutes = exercise.get("cardio_minutes", 20)

    narrative["exercise_plan"] = (
        f"A structured movement program emphasizing approximately {cardio_minutes} minutes "
        f"of cardiovascular activity daily along with mobility and strength training "
        f"can help improve circulation and metabolic function."
    )

    nutrition = protocol.get("nutrition_rules", {})
    foods = nutrition.get("focus_foods", [])

    narrative["nutrition_plan"] = (
        f"Nutritional strategies should emphasize foods such as {', '.join(foods)} "
        f"to support cardiovascular and anti-inflammatory pathways."
    )

    narrative["next_steps"] = (
        "Following the recommended exercise and nutrition strategies consistently "
        "over the next several weeks may help improve physiological balance "
        "and support long-term health outcomes."
    )

    return narrative