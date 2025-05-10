import numpy as np
from dsl import *
from utils.rule_helpers import *

def objects_stretch_to_edges(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False
    
    out_shape = (len(out), len(out[0]))

    for in_obj in inp_objs:
        best_match = None
        max_overlap = 0
        for out_obj in out_objs:
            overlap = len(in_obj & out_obj)
            if overlap > max_overlap:
                best_match = out_obj
                max_overlap = overlap

        if best_match is None:
            continue

        if not touches_edge(in_obj, out_shape) and touches_edge(best_match, out_shape):
            return True

    return False

def object_has_rotated(inp, out, inp_objs=None, out_objs=None):
    from itertools import product

    if not inp_objs or not out_objs:
        return False

    inp_grids = [to_grid(obj) for obj in inp_objs]
    out_grids = [to_grid(obj) for obj in out_objs]

    for inp_grid, out_grid in product(inp_grids, out_grids):
        if any(rot == out_grid for rot in all_rotations(inp_grid)):
            return True  # At least one matching rotation found

    return False  # No rotations found that match

GEOMETRY_RULES = [
    (objects_stretch_to_edges, 1),
    (object_has_rotated, 1)
]
