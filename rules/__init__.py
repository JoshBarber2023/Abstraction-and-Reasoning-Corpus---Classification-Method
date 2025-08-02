from .colour import COLOUR_RULES_NL
from .commonsense import COMMONSENSE_RULES_NL
from .geometry import GEOMETRY_RULES_NL
from .movement import MOVEMENT_RULES_NL
from .number import NUMBER_RULES_NL
from .object import OBJECT_RULES_NL

def normalise_rule_priors(rules):
    total = sum(p for _, p in rules)
    if total == 0:
        raise ValueError("Sum of priors is zero. Cannot normalize.")
    return [(desc, p / total) for desc, p in rules]

ALL_RULES = {
    "Colour": normalise_rule_priors(COLOUR_RULES_NL),
    "CommonSense": normalise_rule_priors(COMMONSENSE_RULES_NL),
    "Geometry": normalise_rule_priors(GEOMETRY_RULES_NL),
    "Movement": normalise_rule_priors(MOVEMENT_RULES_NL),
    "Number": normalise_rule_priors(NUMBER_RULES_NL),
    "Object": normalise_rule_priors(OBJECT_RULES_NL),
}
