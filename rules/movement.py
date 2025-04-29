from dsl import shift, centerofmass, rot90, rot180, rot270
import numpy as np

def check_translation(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return inp != out and len(inp) == len(out)
    return any(centerofmass(o1) != centerofmass(o2) for o1, o2 in zip(sorted(inp_objs), sorted(out_objs)))

def check_rotation(inp, out, inp_objs=None, out_objs=None):
    # Use np.array_equal to compare arrays element-wise
    return (np.array_equal(out, rot90(inp)) or 
            np.array_equal(out, rot180(inp)) or 
            np.array_equal(out, rot270(inp)))

MOVEMENT_RULES = [
    (check_translation, 1),
    (check_rotation, 1),
]