import numpy as np
from dsl import *
from rules.object import objects_get_larger, objects_get_smaller
from utils.rule_helpers import *

def check_object_moved(inp, out, inp_objs=None, out_objs=None):
    """Check if objects moved with a consistent offset, considering size changes."""
    if not inp_objs or not out_objs or len(inp_objs) != len(out_objs):
        return False

    # Check if all objects have just gotten larger, consider them not moved
    if objects_get_larger(inp, out, inp_objs, out_objs):
        return False

    offsets = []
    for in_obj, out_obj in zip(inp_objs, out_objs):
        in_c = centerofmass(in_obj)
        out_c = centerofmass(out_obj)

        # If the object got smaller but the center of mass moved, it's still considered moved
        if objects_get_smaller(inp, out, inp_objs, out_objs):
            if in_c != out_c:
                offsets.append((out_c[0] - in_c[0], out_c[1] - in_c[1]))
        else:
            offsets.append((out_c[0] - in_c[0], out_c[1] - in_c[1]))

    # Check if any object has moved with an offset
    return any(offset != (0, 0) for offset in offsets)

def check_bouncing(inp, out, inp_objs=None, out_objs=None):
    """
    Checks if an object bounces within the grid, either diagonally or vertically.

    Args:
    inp (Grid): The input grid.
    out (Grid): The output grid.
    inp_objs (Objects, optional): The input objects. Defaults to None.
    out_objs (Objects, optional): The output objects. Defaults to None.

    Returns:
    bool: True if the bouncing movement is detected, False otherwise.
    """
    # If there are no input or output objects, return False
    if not inp_objs or not out_objs:
        return False

    # For each object in the input and output, check if their positions have changed
    for inp_obj, out_obj in zip(inp_objs, out_objs):
        # Check that the object has moved (bounced)
        # We'll assume a simple check: positions in the x, y axis either move diagonally or vertically.
        for (inp_cell, out_cell) in zip(inp_obj, out_obj):
            inp_x, inp_y = inp_cell[1]  # Extract x, y from the input object
            out_x, out_y = out_cell[1]  # Extract x, y from the output object

            # Check for diagonal movement (bouncing along diagonal lines)
            if (inp_x != out_x and inp_y != out_y):
                return True

            # Check for vertical bouncing (same x, changing y)
            if inp_x == out_x and inp_y != out_y:
                return True

            # Check for horizontal bouncing (same y, changing x)
            if inp_y == out_y and inp_x != out_x:
                return True

    return False

MOVEMENT_RULES = [
    (check_object_moved, 1),
    (check_bouncing, 1)
]
