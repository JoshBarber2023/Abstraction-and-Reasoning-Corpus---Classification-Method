import numpy as np
from dsl import *

def objects_get_larger(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False

    for in_obj in inp_objs:
        in_coords = {loc for (_, loc) in in_obj}
        in_size = len(in_coords)
        best_match = None
        max_overlap = 0
        for out_obj in out_objs:
            out_coords = {loc for (_, loc) in out_obj}
            overlap = len(in_coords & out_coords)
            if overlap > max_overlap:
                best_match = out_coords
                max_overlap = overlap
        if best_match is None:
            continue
        out_size = len(best_match)
        if out_size <= in_size:
            return False
    return True

def objects_get_smaller(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False

    for in_obj in inp_objs:
        in_coords = {loc for (_, loc) in in_obj}
        in_size = len(in_coords)
        best_match = None
        max_overlap = 0
        for out_obj in out_objs:
            out_coords = {loc for (_, loc) in out_obj}
            overlap = len(in_coords & out_coords)
            if overlap > max_overlap:
                best_match = out_coords
                max_overlap = overlap
        if best_match is None:
            continue
        out_size = len(best_match)
        if out_size >= in_size:
            return False
    return True

OBJECT_RULES = [
    (objects_get_larger, 1),
    (objects_get_smaller, 1)
]
