# ================================
# NATURAL LANGUAGE DESCRIPTIONS
# ================================
NL_RULES = [
    "Only the colours of pixels change between input and output; all pixel positions remain the same.",
    "Each unique colour in the input has exactly one corresponding output colour (1-to-1 mapping).",
    "Objects retain their shape and position, but their colours change consistently across all instances.",
    "No new objects are added or removed — only existing colours are swapped or adjusted.",
    "All changes occur at the pixel level, with colour values altered but spatial layout untouched.",
    "The number of non-background pixels remains constant, confirming no geometric transformation.",
    "The transformation can be fully described by a dictionary mapping input colours to output colours.",
    "Every instance of a specific input colour changes to the same output colour globally.",
    "There is no spatial dependency — the same colour transformation applies regardless of position.",
    "Background colour (most common colour) remains the same in input and output grids."
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