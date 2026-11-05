"""Rule-of-thumb maintenance text. This is not a certified diagnosis."""


def maintenance_guidance(condition: str, soh: float) -> list[str]:
    notes = [
        "These notes follow the predicted laboratory SoH band. They are not a warranty decision, a roadworthiness result, or a substitute for a capacity test."
    ]
    if condition == "Poor" or soh <= 70:
        notes.append(
            "The estimate is in the Poor band used by this project (at or below 70% of the 2.0 Ah reference). Plan a service inspection and do not treat the vehicle range as reliable until capacity is measured."
        )
    elif condition == "Fair" or soh <= 85:
        notes.append(
            "The estimate is in the Fair band (above 70% and at or below 85%). Shorten the inspection interval and avoid repeated deep discharge while the result is confirmed with a proper capacity measurement."
        )
    else:
        notes.append(
            "The estimate is in the Good band (above 85% of the 2.0 Ah reference). Continue routine monitoring. This is still a model estimate from laboratory-style inputs."
        )
    notes.append(
        "If the entered values came from a vehicle pack rather than a single-cell discharge, discard this estimate. The model was not trained on pack voltage or pack current."
    )
    return notes
