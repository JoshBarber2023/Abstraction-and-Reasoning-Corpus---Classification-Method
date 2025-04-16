import numpy as np

def check_new_object_created(inp, out):
    """
    Checks if a new object has been created in the output grid.
    This rule identifies the creation of new non-zero elements.
    """
    return not np.array_equal(inp, out) and np.any(out > 0)

def check_object_transformed(inp, out):
    """
    Checks if an object has been transformed (its pixel content changed) but the overall structure remains.
    The sorted content of the object remains the same, possibly rearranged.
    """
    return np.array_equal(np.sort(inp, axis=None), np.sort(out, axis=None)) and not np.array_equal(inp, out)

OBJECT_RULES = [
    (check_new_object_created, 1),
    (check_object_transformed, 1)
]
