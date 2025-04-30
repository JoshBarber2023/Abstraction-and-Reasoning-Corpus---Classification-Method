import numpy as np
from dsl import normalize, rot90, rot180, rot270, toobject, shape, crop, centerofmass


def check_translation(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs or len(inp_objs) != len(out_objs):
        return not np.array_equal(inp, out) and np.array(inp).shape == np.array(out).shape
    return any(centerofmass(in_obj) != centerofmass(out_obj)
               for in_obj, out_obj in zip(sorted(inp_objs), sorted(out_objs)))


def check_rotation(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return False

    norm_out_objs = [normalize(obj) for obj in out_objs]

    for in_obj in inp_objs:
        norm_in_obj = normalize(in_obj)
        in_grid = crop(inp, *shape(norm_in_obj))
        candidates = [
            toobject(normalize(rot90(in_grid)), out),
            toobject(normalize(rot180(in_grid)), out),
            toobject(normalize(rot270(in_grid)), out),
        ]
        for rotated in candidates:
            if normalize(rotated) in norm_out_objs:
                return True
    return False

MOVEMENT_RULES = [
    (check_translation, 1),
    (check_rotation, 1),
]
