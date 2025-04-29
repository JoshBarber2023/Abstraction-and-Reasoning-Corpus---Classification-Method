import numpy as np
from dsl import lowermost, backdrop

def check_gravity_down(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return inp.sum() < out.sum()
    return all(lowermost(obj_out) >= lowermost(obj_in) for obj_in, obj_out in zip(sorted(inp_objs), sorted(out_objs)))

def check_containment_change(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return inp.shape == out.shape and np.any(inp != out)

    # Safely handle empty object sets
    inp_objs = sorted(inp_objs) if inp_objs else []
    out_objs = sorted(out_objs) if out_objs else []

    return any(obj_in != obj_out for obj_in, obj_out in zip(inp_objs, out_objs))

def check_ring_filling(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return np.any((inp == 0) & (out != 0))
    for obj_in in inp_objs:
        area = backdrop(obj_in)
        if any(out[i][j] != 0 for i, j in area if inp[i][j] == 0):
            return True
    return False

COMMONSENSE_RULES = [
    (check_gravity_down, 1),
    (check_containment_change, 1),
    (check_ring_filling, 1)
]
