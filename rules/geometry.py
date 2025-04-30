import numpy as np
from dsl import normalize

def check_horizontal_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.flipud(out))

def check_vertical_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.fliplr(out))

def check_shapes_preserved(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return len(np.unique(inp[inp != 0])) == len(np.unique(out[out != 0]))
    return all(normalize(o1) in [normalize(o2) for o2 in out_objs] for o1 in inp_objs)

GEOMETRY_RULES = [
    (check_horizontal_symmetry, 0.5),
    (check_vertical_symmetry, 0.5),
    (check_shapes_preserved, 1),
]
