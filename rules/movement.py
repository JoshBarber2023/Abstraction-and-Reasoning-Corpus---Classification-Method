import numpy as np
from dsl import *

def check_translation(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs or len(inp_objs) != len(out_objs):
        return not np.array_equal(inp, out) and np.array(inp).shape == np.array(out).shape
    return any(centerofmass(in_obj) != centerofmass(out_obj)
               for in_obj, out_obj in zip(sorted(inp_objs), sorted(out_objs)))

def check_object_swapping(inp, out, inp_objs=None, out_objs=None):
    """Two objects have swapped locations."""
    if len(inp_objs) != 2 or len(out_objs) != 2:
        return False
    return sorted(centerofmass(o) for o in inp_objs) == sorted(centerofmass(o) for o in out_objs)[::-1]

def check_slide_to_edge(inp, out, inp_objs=None, out_objs=None):
    """Objects moved toward the closest boundary."""
    if not inp_objs or not out_objs:
        return False
    for obj_out in out_objs:
        if isinstance(out, np.ndarray):
            max_dim = out.shape[0]
        else:
            max_dim = len(out)
        if min(centerofmass(obj_out)) != 0 and max(centerofmass(obj_out)) != max_dim - 1:
            return False
    return True

def check_downscale(inp, out, inp_objs=None, out_objs=None):
    """Check if the grid was downscaled by uniform factor."""
    in_height, in_width = len(inp), len(inp[0])
    out_height, out_width = len(out), len(out[0])

    if in_height % out_height != 0 or in_width % out_width != 0:
        return False

    factor_h = in_height // out_height
    factor_w = in_width // out_width

    for i in range(out_height):
        for j in range(out_width):
            if inp[i * factor_h][j * factor_w] != out[i][j]:
                return False
    return True

def check_move_backtrack(inp, out, inp_objs=None, out_objs=None):
    """Check if all objects moved with a consistent offset."""
    if not inp_objs or not out_objs or len(inp_objs) != len(out_objs):
        return False

    offsets = []
    for in_obj, out_obj in zip(inp_objs, out_objs):
        in_c = centerofmass(in_obj)
        out_c = centerofmass(out_obj)
        offsets.append((out_c[0] - in_c[0], out_c[1] - in_c[1]))

    return all(offset == offsets[0] for offset in offsets)

MOVEMENT_RULES = [
    (check_translation, 1),
    (check_object_swapping, 1),
    (check_slide_to_edge, 1),
    (check_downscale, 1),
    (check_move_backtrack, 1),
]
