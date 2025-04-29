import numpy as np

def check_new_object_created(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return np.any(out > inp)
    return len(out_objs) > len(inp_objs)

def check_object_transformed(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return np.array_equal(np.sort(inp, axis=None), np.sort(out, axis=None)) and not np.array_equal(inp, out)
    for obj_in in inp_objs:
        matched = False
        for obj_out in out_objs:
            if sorted([v for v, _ in obj_in]) == sorted([v for v, _ in obj_out]):
                matched = True
                break
        if not matched:
            return True
    return False

OBJECT_RULES = [
    (check_new_object_created, 1),
    (check_object_transformed, 1)
]
