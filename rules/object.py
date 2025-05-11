import numpy as np
from dsl import *
from utils.rule_helpers import *
from rules.commonsense import *

def objects_get_larger(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False
    
    if shape_to_color_relationship(inp, out, inp_objs, out_objs): 
        return  False

    # Convert each object to a set of locations for comparison
    in_obj_sizes = [len({loc for _, loc in in_obj}) for in_obj in inp_objs]
    out_obj_sizes = [len({loc for _, loc in out_obj}) for out_obj in out_objs]

    # Check if any output object is larger than any input object
    for out_size in out_obj_sizes:
        for in_size in in_obj_sizes:
            if out_size > in_size:
                return True

    return False

def objects_get_smaller(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False
    
    if shape_to_color_relationship(inp, out, inp_objs, out_objs): 
        return  False

    # Convert each object to a set of locations for comparison
    in_obj_sizes = [len({loc for _, loc in in_obj}) for in_obj in inp_objs]
    out_obj_sizes = [len({loc for _, loc in out_obj}) for out_obj in out_objs]

    # Check if any output object is smaller than any input object
    for out_size in out_obj_sizes:
        for in_size in in_obj_sizes:
            if out_size < in_size:
                return True

    return False

def neighbour_object_disappears(inp, out, inp_objs=None, out_objs=None):
    from rules.colour import all_objects_change_colour  # Moved here to avoid circular import
    if inp_objs is None or out_objs is None:
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

OBJECT_RULES = [
    (objects_get_larger, 1),
    (objects_get_smaller, 1),
    (neighbour_object_disappears, 1),
    (neighbour_object_appears, 1)
]
