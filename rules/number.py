import numpy as np

def check_pixel_count_change(inp, out, inp_objs=None, out_objs=None):
    return np.count_nonzero(inp) != np.count_nonzero(out)

def check_object_count_change(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return np.unique(inp).size != np.unique(out).size
    return len(inp_objs) != len(out_objs)

NUMBER_RULES = [
    (check_pixel_count_change, 1),
    (check_object_count_change, 1)
]