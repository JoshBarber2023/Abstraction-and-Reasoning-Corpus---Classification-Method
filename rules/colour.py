import numpy as np
from dsl import color

def check_colour_mapping(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return np.unique(inp).size != np.unique(out).size
    return len(set(color(obj) for obj in inp_objs)) != len(set(color(obj) for obj in out_objs))

def check_palette_swap(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return sorted(np.unique(inp)) != sorted(np.unique(out))
    in_palette = sorted(set(color(obj) for obj in inp_objs))
    out_palette = sorted(set(color(obj) for obj in out_objs))
    return in_palette != out_palette

COLOUR_RULES = [
    (check_colour_mapping, 1),
    (check_palette_swap, 1),
]