import numpy as np

def check_horizontal_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.flipud(out))

def check_vertical_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.fliplr(out))

def check_shapes_preserved(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return len(np.unique(inp[inp != 0])) == len(np.unique(out[out != 0]))
    return len(inp_objs) == len(out_objs)

GEOMETRY_RULES = [
    (check_horizontal_symmetry, 1),
    (check_vertical_symmetry, 1),
    (check_shapes_preserved, 1),
]
