import numpy as np
from dsl import normalize

def check_horizontal_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.flipud(out)) and np.array_equal(inp, np.flipud(inp))

def check_vertical_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.fliplr(out)) and np.array_equal(inp, np.fliplr(inp))

def check_shapes_preserved(inp, out, inp_objs=None, out_objs=None):
    inp = np.array(inp)
    out = np.array(out)

    if not inp_objs or not out_objs:
        return len(np.unique(inp[inp != 0])) == len(np.unique(out[out != 0]))
    
    return all(normalize(o1) in [normalize(o2) for o2 in out_objs] for o1 in inp_objs)

def check_shape_mirroring(inp, out, inp_objs=None, out_objs=None):
    """Input is mirrored across vertical/horizontal axis."""
    return np.array_equal(out, np.flip(inp, axis=1)) or np.array_equal(out, np.flip(inp, axis=0))

GEOMETRY_RULES = [
    (check_horizontal_symmetry, 1),
    (check_vertical_symmetry, 1),
    (check_shapes_preserved, 1),
    (check_shape_mirroring, 1),
]
