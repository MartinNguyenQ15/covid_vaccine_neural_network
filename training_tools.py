all_categories = {
    'covid19_vaccination': [1, 0],
    'safety_side_effects': [1, 0],
    'vaccination_intent': [0, 1],
}

def phrase_scoring(x=""):
    inp = x.lower()
    score = 0
    intent = 0
    kws = 0
    keywords = [
        "covid",
        "vaccine",
        "moderna",
        "pfizer",
        "johnson",
        "cdc",
        "department",
        "health",
    ]
    for k in keywords:
        if k in inp:
            score += 1
            kws += 1
    lowering_keywords = [
        "cvs",
        "walmart",
        "walgreens",
        "when",
        "near",
        "me",
        "where",
        "rite aid"
    ]
    for k in lowering_keywords:
        if k in inp:
            intent += 1
            kws += 1
    if kws == 0:
        return (0, 0)
    return (score / kws, intent / kws)