import importlib
import pkgutil
import sys

def normalise_rule_priors(rules):
    # If rules is a list of functions without priors, assign default prior=1.0
    if rules and not isinstance(rules[0], tuple):
        rules = [(r, 1.0) for r in rules]

    total = sum(p for _, p in rules)
    if total == 0:
        raise ValueError("Sum of priors is zero. Cannot normalize.")
    return [(func, p / total) for func, p in rules]


ALL_RULES = {}

# Dynamically load all modules in this package
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    # Import the module, e.g. 'rules.colour'
    full_module_name = f"{__name__}.{module_name}"
    module = importlib.import_module(full_module_name)
    
    # Expect each module to have a variable named <MODULE_NAME_UPPER>_RULES
    # E.g. colour.py -> COLOUR_RULES
    attr_name = f"{module_name.upper()}_RULES"
    
    if hasattr(module, attr_name):
        rules = getattr(module, attr_name)
        ALL_RULES[module_name.capitalize()] = normalise_rule_priors(rules)
    else:
        print(f"Warning: Module '{module_name}' does not define '{attr_name}'")

# Optionally expose ALL_RULES at package level
__all__ = ["ALL_RULES"]
