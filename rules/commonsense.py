import numpy as np
from dsl import *
import math
from utils.rule_helpers import *

def check_filled_relationship(inp, out, inp_objs=None, out_objs=None):
    """
    Check if background inside/around objects was filled or cleared.
    """
    inp = np.array(inp)
    out = np.array(out)

    if not inp_objs:
        return np.any(inp != out)

    out_h, out_w = out.shape
    for obj_in in inp_objs:
        for i, j in backdrop(obj_in):
            if 0 <= i < out_h and 0 <= j < out_w:
                if inp[i][j] != out[i][j]:
                    return True
    return False

def check_column_ascending_order(inp, out, inp_objs=None, out_objs=None):
    if out_objs is None:
        return False

    max_x = len(out)
    max_y = len(out[0]) if len(out) > 0 else 0

    for obj in out_objs:
        if not is_vertically_aligned(obj):  # Check if vertically aligned
            return False  # Object is "floating" or misaligned

        sorted_cells = sorted(obj, key=lambda c: c[0])  # Sort top to bottom
        values = [out[x][y] for x, y in sorted_cells if 0 <= x < max_x and 0 <= y < max_y]

        if values != sorted(values):
            return False

    return True

def check_column_descending_order(inp, out, inp_objs=None, out_objs=None):
    if out_objs is None:
        return False

    max_x = len(out)
    max_y = len(out[0]) if len(out) > 0 else 0

    for obj in out_objs:
        if not is_vertically_aligned(obj):  # Check if vertically aligned
            return False  # Object is "floating" or misaligned

        sorted_cells = sorted(obj, key=lambda c: c[0])  # Sort top to bottom
        values = [out[x][y] for x, y in sorted_cells if 0 <= x < max_x and 0 <= y < max_y]

        if values != sorted(values, reverse=True):
            return False

    return True

def check_row_ascending_order(inp, out, inp_objs=None, out_objs=None):
    if out_objs is None:
        return False

    max_x = len(out)
    max_y = len(out[0]) if len(out) > 0 else 0

    for obj in out_objs:
        if not is_horizontally_aligned(obj):  # Check if horizontally aligned
            return False  # Object is "floating" or misaligned

        sorted_cells = sorted(obj, key=lambda c: c[1])  # Sort left to right
        values = [out[x][y] for x, y in sorted_cells if 0 <= x < max_x and 0 <= y < max_y]

        if values != sorted(values):
            return False

    return True

def check_row_descending_order(inp, out, inp_objs=None, out_objs=None):
    if out_objs is None:
        return False

    max_x = len(out)
    max_y = len(out[0]) if len(out) > 0 else 0

    for obj in out_objs:
        if not is_horizontally_aligned(obj):  # Check if horizontally aligned
            return False  # Object is "floating" or misaligned

        sorted_cells = sorted(obj, key=lambda c: c[1])  # Sort left to right
        values = [out[x][y] for x, y in sorted_cells if 0 <= x < max_x and 0 <= y < max_y]

        if values != sorted(values, reverse=True):
            return False

    return True

def shape_to_color_relationship(inp, out, inp_objs=None, out_objs=None):
    """
    Check if the relationship between the input and output objects follows
    the pattern where the color of the objects changes based on shape, with
    some matching pixel to the input object in the output.

    :param inp: The input grid.
    :param out: The output grid.
    :param inp_objs: The input objects (optional).
    :param out_objs: The output objects (optional).
    :return: True if the relationship matches the described behavior, False otherwise.
    """

    from rules.geometry import objects_stretch_to_edges

    if objects_stretch_to_edges(inp, out, inp_objs, out_objs):
        return False

    # Ensure object sets exist
    if inp_objs is None or out_objs is None:
        return False
    
    # If there's only one object in both input and output, skip
    if len(inp_objs) != len(out_objs):
        return False

    # If there's only one object in both input and output, skip
    if len(inp_objs) == 1 and len(out_objs) == 1:
        return False

    # Extract pixels for the input and output objects
    input_pixels = {cell[1] for obj in inp_objs for cell in obj}
    output_pixels = {cell[1] for obj in out_objs for cell in obj}

    # Check if there is at least one matching pixel between the input and output
    if input_pixels & output_pixels:  # If there's any intersection between input and output pixels
        # Now we check if the color has changed but some part of the shape remains
        return True
    return False

def combined_objects_form_common_shape(inp, out, inp_objs: Objects = None, out_objs: Objects = None) -> bool:
    if inp_objs is None or out_objs is None:
        return False

    for in_obj in inp_objs:
        for out_obj in out_objs:
            color1 = get_object_color(in_obj)
            color2 = get_object_color(out_obj)

            # Only combine if both have the same single color
            if color1 is None or color2 is None or color1 != color2:
                continue

            # Check for overlap — this returns True if they share any pixel
            if objects_overlap(in_obj, out_obj):
                continue  # Skip this pair if they overlap

            combined = combine_objects(in_obj, out_obj)

            if is_square(combined) or is_rectangle(combined) or is_circle(combined) or is_triangle(combined):
                return True

    return False

def check_checkerboard_pattern(inp, out, inp_objs=None, out_objs=None):
    """
    Checks if the output grid is a checkerboard pattern.
    
    Args:
        inp (Grid): The input grid.
        out (Grid): The output grid.
        inp_objs (Objects, optional): Objects in the input grid.
        out_objs (Objects, optional): Objects in the output grid.
    
    Returns:
        Boolean: True if the output grid forms a checkerboard pattern, False otherwise.
    """
    
    rows = len(out)
    cols = len(out[0])

    # Check if the output grid forms a checkerboard pattern
    for r in range(rows):
        for c in range(cols):
            # Check if alternating pattern holds
            if (r + c) % 2 == 0:
                if out[r][c] != out[0][0]:  # Compare with the top-left corner
                    return False
            else:
                if out[r][c] == out[0][0]:  # Compare with the top-left corner
                    return False
    return True

def outputs_do_not_overlap_inputs(inp, out, inp_objs=None, out_objs=None) -> bool:
    """
    Returns True if none of the output objects overlap with any input object.
    """
    if inp_objs is None or out_objs is None:
        return False

    for out_obj in out_objs:
        if any(not out_obj.isdisjoint(in_obj) for in_obj in inp_objs):
            # This output object overlaps with an input object
            return False

    return True

def check_tetris_relationship(inp, out, inp_objs: Objects = None, out_objs: Objects = None) -> bool:
    """
    Detects Tetris-like behavior between inp_objs and out_objs, but ONLY if
    there are objects on the bottom row in the output.

    Tetris-like behavior includes:
    - Rows cleared (some row from input is missing in output),
    - Object moved down (same shape, lower x),
    - Gravity applied (object shifted to lowest possible position).

    Args:
        inp, out: Ignored (retained for compatibility).
        inp_objs (Objects): Input objects (frozensets of (val, (x, y))).
        out_objs (Objects): Output objects.

    Returns:
        bool: True if Tetris-like transformation is detected AND object touches bottom row.
    """
    if inp_objs is None or out_objs is None or not out_objs:
        return False

    def extract_coords(obj: Object):
        return {(x, y) for _, (x, y) in obj}

    def get_rows(objs: Objects):
        return {x for obj in objs for _, (x, _) in obj}

    def normalize(obj_coords):
        """ Normalize shape to top-left origin """
        min_x = min(x for x, y in obj_coords)
        min_y = min(y for x, y in obj_coords)
        return {(x - min_x, y - min_y) for x, y in obj_coords}

    # Determine bottom-most y-coordinate in output (max row index)
    out_max_y = max(y for obj in out_objs for _, (_, y) in obj)

    # Require at least one object cell on bottom row in output
    if any(y == out_max_y for obj in out_objs for _, (_, y) in obj):
        return False


    # 1. Check for row clearance
    inp_rows = get_rows(inp_objs)
    out_rows = get_rows(out_objs)
    cleared_rows = inp_rows - out_rows
    if cleared_rows:
        return True

    # 2. Check for downward movement (same shape, lower in output)
    out_coords_sets = [extract_coords(o) for o in out_objs]
    in_coords_sets = [extract_coords(o) for o in inp_objs]

    for in_obj in in_coords_sets:
        in_shape = normalize(in_obj)
        in_min_x = min(x for x, y in in_obj)

        for out_obj in out_coords_sets:
            out_shape = normalize(out_obj)
            out_min_x = min(x for x, y in out_obj)

            if in_shape == out_shape and out_min_x > in_min_x:
                return True  # Same shape moved down

    # 3. Gravity: check if objects have sunk fully (i.e. no floating blocks)
    for out_obj in out_coords_sets:
        below_cells = {(x + 1, y) for x, y in out_obj}
        if all((x, y) not in out_obj for (x, y) in below_cells):
            return True  # Object rests on solid ground

    return False


COMMONSENSE_RULES = [
    (check_filled_relationship, 1),
    (check_column_ascending_order, 1),
    (check_column_descending_order, 1),
    (shape_to_color_relationship, 1),
    (combined_objects_form_common_shape, 1),
    (check_checkerboard_pattern, 1),
    (outputs_do_not_overlap_inputs, 1),
    (check_tetris_relationship, 1)
]