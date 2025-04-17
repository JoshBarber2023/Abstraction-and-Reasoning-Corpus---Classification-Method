import numpy as np

def check_horizontal_symmetry(inp, out):
    return np.array_equal(out, np.flipud(out))

def check_vertical_symmetry(inp, out):
    return np.array_equal(out, np.fliplr(out))

def check_shapes_preserved(inp, out):
    return len(np.unique(inp[inp != 0])) == len(np.unique(out[out != 0]))

GEOMETRY_RULES = [
    (check_horizontal_symmetry, 1),
    (check_vertical_symmetry, 1),
    (check_shapes_preserved, 1),
]

