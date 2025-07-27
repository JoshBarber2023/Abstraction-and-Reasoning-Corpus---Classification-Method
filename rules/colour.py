# ================================
# NATURAL LANGUAGE DESCRIPTIONS
# ================================
NL_RULES = [
    "The transformation involves only changes to colours; object shapes and positions are preserved.",
    "All colour regions are retained — no merging, splitting, or reshaping of coloured areas.",
    "The output colours are a direct mapping or swap of the input palette, with no new colours introduced.",
    "Partial internal colour changes occur only when smaller objects appear adjacent to existing ones without changing object shapes or positions."
]

# ==================================
# SIMPLE FUNDAMENTAL COLOUR CHECKS
# ==================================

from dsl import *
import numpy as np

def all_objects_change_colour(inp, out, inp_objs=None, out_objs=None) -> bool:
    """
    Returns True if every object in the input has changed its colour in the output,
    assuming objects maintain identity via shape and position.
    """
    if inp_objs is None or out_objs is None:
        return False

    # Match input and output objects by shape and position
    matched_pairs = []
    for in_obj in inp_objs:
        for out_obj in out_objs:
            if {cell[1] for cell in in_obj} == {cell[1] for cell in out_obj}:  # match by shape/position
                matched_pairs.append((in_obj, out_obj))
                break

    if len(matched_pairs) != len(inp_objs):
        return False  # not all objects could be matched

    # Check if all matched objects changed colour
    for in_obj, out_obj in matched_pairs:
        in_color = next(iter(in_obj))[0]
        out_color = next(iter(out_obj))[0]
        if in_color == out_color:
            return False

    return True

def mimic_colour_scheme(inp, out, inp_objs=None, out_objs=None):
    """
    This function checks if the output objects have mimicked the colour scheme of the input objects.
    It does so by comparing the colours of the input and output objects and ensuring they match,
    but only if more than two distinct colours exist in the grid.
    """
    if inp_objs is None or out_objs is None:
        # Can't proceed without objects information
        return False

    # Flatten input and output grids to count distinct colours
    input_colours = set(cell for row in inp for cell in row)
    output_colours = set(cell for row in out for cell in row)

    # Combined colour palette from input and output
    total_colours = input_colours | output_colours

    # If there are 3 or fewer colours, skip this rule
    if len(total_colours) <= 3:
        return False

    # Check if the number of objects is the same
    if len(inp_objs) != len(out_objs):
        return False

    # Check if each object’s primary colour is preserved
    for inp_obj, out_obj in zip(inp_objs, out_objs):
        inp_col = next(iter(inp_obj))[0]
        out_col = next(iter(out_obj))[0]
        if inp_col != out_col:
            return False

    return True

def partial_internal_colour_change(inp, out, inp_objs=None, out_objs=None):
    from rules.Object import objects_get_smaller, neighbour_object_appears
    
    if inp_objs is None or out_objs is None:
        return False

    if objects_get_smaller(inp, out, inp_objs, out_objs) and neighbour_object_appears(inp, out, inp_objs, out_objs):
        return True

    return False

COLOUR_RULES = [
    (all_objects_change_colour, 1),
    (mimic_colour_scheme, 1),
    (partial_internal_colour_change, 1)
]