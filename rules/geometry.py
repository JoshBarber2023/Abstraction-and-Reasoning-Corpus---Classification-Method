import numpy as np
from dsl import *
from utils.rule_helpers import *
from rules.object import *

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

    if objects_get_larger(inp, out, inp_objs, out_objs) or objects_get_smaller(inp, out, inp_objs, out_objs) or shape_to_color_relationship(inp, out, inp_objs, out_objs):
        return False

    inp_grids = [to_grid(obj) for obj in inp_objs]
    out_grids = [to_grid(obj) for obj in out_objs]

    def is_rotationally_symmetric(grid):
        # Check if grid is the same under all 90-degree rotations
        return all(rot == grid for rot in all_rotations(grid))

    for inp_grid in inp_grids:
        # Skip objects that can't visually rotate
        if sum(cell is not None for row in inp_grid for cell in row) == 1:
            continue  # single pixel
        if is_rotationally_symmetric(inp_grid):
            continue  # e.g., square symmetric object

        matched = False
        for out_grid in out_grids:
            if any(rot == out_grid for rot in all_rotations(inp_grid)):
                matched = True
                break
        if not matched:
            return True

    return False

def is_completely_surrounded(inp, out, inp_objs=None, out_objs=None): 
    # Ensure input objects and output objects are provided
    if not out_objs or not inp_objs:
        return False

    for out_obj in out_objs:
        for inp_obj in inp_objs:
            # Check if an object completely surrounds another by verifying the boundary points
            surrounded = True
            for cell in inp_obj:
                # Check if each cell in the input object is within the boundaries of the output object
                if cell not in out_obj:
                    surrounded = False
                    break
            if surrounded:
                return True

    return False

GEOMETRY_RULES = [
    (objects_stretch_to_edges, 1),
    (object_has_rotated, 1),
    (is_completely_surrounded, 1)
]
