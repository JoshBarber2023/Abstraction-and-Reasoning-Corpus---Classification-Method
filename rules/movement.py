import numpy as np
from scipy.ndimage import label

def check_translation(inp, out):
    """
    Checks if the grid has been translated, meaning the position of elements shifted 
    without resizing or rotation. The grid shape remains the same, but the positions of the pixels 
    have changed.
    """
    return inp.shape == out.shape and not np.array_equal(inp, out)

def check_rotation(inp, out, angle=90):
    """
    Checks if the grid has been rotated by a specified angle (e.g., 90, 180, 270 degrees).
    The output grid will be compared against a rotated version of the input.
    """
    if angle == 90:
        return np.array_equal(np.rot90(inp), out)
    elif angle == 180:
        return np.array_equal(np.rot90(np.rot90(inp)), out)
    elif angle == 270:
        return np.array_equal(np.rot90(np.rot90(np.rot90(inp))), out)
    else:
        raise ValueError("Only 90, 180, and 270 degree rotations are supported.")

MOVEMENT_RULES = [
    (check_translation, 1),
    (check_rotation, 1),
]
