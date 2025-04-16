import numpy as np

def check_colour_mapping(inp, out):
    return np.unique(inp).size != np.unique(out).size

def check_palette_swap(inp, out):
    return sorted(np.unique(inp)) != sorted(np.unique(out))

COLOUR_RULES = [
    (check_colour_mapping, 1),
    (check_palette_swap, 1),
]
