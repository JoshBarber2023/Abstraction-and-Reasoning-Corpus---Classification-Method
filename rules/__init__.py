from .colour import COLOUR_RULES
from .commonsense import COMMONSENSE_RULES
from .geometry import GEOMETRY_RULES
from .movement import MOVEMENT_RULES
from .number import NUMBER_RULES
from .object import OBJECT_RULES

def normalise_rule_priors(rules):
    total = sum(p for _, p in rules)
    if total == 0:
        raise ValueError("Sum of priors is zero. Cannot normalize.")
    return [(func, p / total) for func, p in rules]

ALL_RULES = {
    "Colour": normalise_rule_priors(COLOUR_RULES),
    "CommonSense": normalise_rule_priors(COMMONSENSE_RULES),
    "Geometry": normalise_rule_priors(GEOMETRY_RULES),
    "Movement": normalise_rule_priors(MOVEMENT_RULES),
    "Number": normalise_rule_priors(NUMBER_RULES),
    "Object": normalise_rule_priors(OBJECT_RULES),
}
