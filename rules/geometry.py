import numpy as np

def check_horizontal_symmetry(inp, out):
    """
    Checks if the input and output grids exhibit horizontal symmetry (mirror reflection).
    """
    return np.array_equal(inp, np.flip(out, axis=1))

def check_vertical_symmetry(inp, out):
    """
    Checks if the input and output grids exhibit vertical symmetry (mirror reflection).
    """
    return np.array_equal(inp, np.flip(out, axis=0))

GEOMETRY_RULES = [
    (check_horizontal_symmetry, 1),
    (check_vertical_symmetry, 1),
]

