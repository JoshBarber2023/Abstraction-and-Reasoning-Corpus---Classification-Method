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
    inp = np.array(inp)
    out = np.array(out)

    if not inp_objs:
        return np.any((inp == 0) & (out != 0))
    
    out_h, out_w = out.shape
    for obj_in in inp_objs:
        for i, j in backdrop(obj_in):
            if 0 <= i < out_h and 0 <= j < out_w:
                if inp[i][j] == 0 and out[i][j] != 0:
                    return True
    return False

def check_intuitive_completion(inp, out, inp_objs=None, out_objs=None):
    """Output completes a partial pattern seen in input."""
    inp = np.array(inp)
    out = np.array(out)

    if inp.shape != out.shape:
        return False

    return np.count_nonzero(inp) < np.count_nonzero(out) and np.any(inp != out)

COMMONSENSE_RULES = [
    (check_gravity_down, 1),
    (check_containment_change, 1),
    (check_ring_filling, 1),
    (check_intuitive_completion, 1),
]
