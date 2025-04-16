import numpy as np

def check_pixel_count_change(inp, out):
    """
    Checks if the number of non-zero pixels has changed between the input and output.
    This could indicate a transformation that adds or removes pixels.
    """
    return np.count_nonzero(inp) != np.count_nonzero(out)

def check_object_count_change(inp, out):
    """
    Checks if the number of unique objects (groups of adjacent pixels) has changed.
    This may suggest that objects were created, merged, or deleted.
    """
    return np.unique(inp).size != np.unique(out).size

NUMBER_RULES = [
    (check_pixel_count_change, 1),
    (check_object_count_change, 1)
]
