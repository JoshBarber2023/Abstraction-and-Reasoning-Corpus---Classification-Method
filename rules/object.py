import numpy as np
from dsl import *
from utils.rule_helpers import *
from rules.commonsense import *
from rules.geometry import *

def objects_get_larger(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False

    if object_removed_but_gap_remains(inp, out, inp_objs, out_objs):
        return False

    if shape_to_color_relationship(inp, out, inp_objs, out_objs): 
        return False

    matches = match_objects_by_overlap(inp_objs, out_objs)

    for in_obj, out_obj in matches:
        if len(out_obj) > len(in_obj):
            return True

    return False

def objects_get_smaller(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False

    if object_removed_but_gap_remains(inp, out, inp_objs, out_objs):
        return False

    if shape_to_color_relationship(inp, out, inp_objs, out_objs): 
        return False

    matches = match_objects_by_overlap(inp_objs, out_objs)

    for in_obj, out_obj in matches:
        if len(out_obj) < len(in_obj):
            return True

    return False

def neighbour_object_disappears(inp, out, inp_objs=None, out_objs=None):
    from rules.colour import all_objects_change_colour  # Moved here to avoid circular import
    if inp_objs is None or out_objs is None:
        return False
    
    # Early exit if number of objects hasn't changed
    if len(inp_objs) == len(out_objs):
        return False
    
    if all_objects_change_colour(inp, out, inp_objs, out_objs) or shape_to_color_relationship(inp, out, inp_objs, out_objs):
        return False

    removed_objs = inp_objs - out_objs
    if not removed_objs:
        return False  # No object disappeared

    for removed in removed_objs:
        for other in inp_objs:
            if other == removed:
                continue
            if objects_are_neighbours(removed, other):
                return True  # Found a neighbour pair where one disappears

    return False

def neighbour_object_appears(inp, out, inp_objs=None, out_objs=None):
    from rules.colour import all_objects_change_colour  # Moved here to avoid circular import
    if inp_objs is None or out_objs is None:
        return False
    
    # Early exit if number of objects hasn't changed
    if len(inp_objs) == len(out_objs):
        return False
    
    if all_objects_change_colour(inp, out, inp_objs, out_objs) or shape_to_color_relationship(inp, out, inp_objs, out_objs):
        return False

    added_objs = out_objs - inp_objs
    if not added_objs:
        return False  # No object appeared

    for added in added_objs:
        for other in out_objs:
            if other == added:
                continue
            if objects_are_neighbours(added, other):
                return True  # Found a neighbour pair where one appears

    return False

def check_object_duplication(inp, out, inp_objs=None, out_objs=None):
    """
    Check if an input object has been copied directly over to the output grid
    and repeated by some ratio or relationship.

    Parameters:
    - inp (Grid): The input grid.
    - out (Grid): The output grid.
    - inp_objs (Objects, optional): The input objects (discrete pixel groups) to check.
    - out_objs (Objects, optional): The output objects (discrete pixel groups) to check.
    
    Returns:
    - Boolean: True if duplication or repetition of the input object is detected in the output, False otherwise.
    """
    # If input objects are not provided, use inp_objs as a fallback
    inp_objs = inp_objs or {frozenset(cell) for cell in inp}

    # If output objects are not provided, use out_objs as a fallback
    out_objs = out_objs or {frozenset(cell) for cell in out}

    # Check if the input object is duplicated in the output grid
    for obj in inp_objs:
        if obj in out_objs:
            # Check the ratio or repeated instances of the input object
            repeated_count = sum(1 for o in out_objs if o == obj)
            if repeated_count > 1:
                return True

    return False

OBJECT_RULES = [
    (objects_get_larger, 1),
    (objects_get_smaller, 1),
    (neighbour_object_disappears, 1),
    (neighbour_object_appears, 1)
]
