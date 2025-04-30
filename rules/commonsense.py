import numpy as np
from dsl import lowermost, backdrop, toindices

def check_gravity_down(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs or len(inp_objs) != len(out_objs):
        return np.array(inp).sum() < np.array(out).sum()
    return any(lowermost(out_obj) > lowermost(in_obj)
               for in_obj, out_obj in zip(sorted(inp_objs), sorted(out_objs)))

def check_containment_change(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return np.any(inp != out)
    return any(toindices(in_obj) != toindices(out_obj)
               for in_obj, out_obj in zip(sorted(inp_objs), sorted(out_objs)))

def check_ring_filling(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs:
        return np.any((inp == 0) & (out != 0))
    for obj_in in inp_objs:
        for i, j in backdrop(obj_in):
            if inp[i][j] == 0 and out[i][j] != 0:
                return True
    return False

COMMONSENSE_RULES = [
    (check_gravity_down, 1),
    (check_containment_change, 1),
    (check_ring_filling, 1)
]
