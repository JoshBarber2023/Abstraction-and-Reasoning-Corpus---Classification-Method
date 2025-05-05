import numpy as np
from dsl import normalize, rot90, rot180, rot270, centerofmass

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
        # Ensure out is a NumPy array before accessing .shape, or use len(out) if it's a tuple
        if isinstance(out, np.ndarray):
            max_dim = out.shape[0]
        else:
            max_dim = len(out)
        
        if min(centerofmass(obj_out)) != 0 and max(centerofmass(obj_out)) != max_dim - 1:
            return False
    return True

def check_rotation(inp, out, inp_objs=None, out_objs=None):
    """Check if any object or the whole grid has been rotated."""
    inp = np.array(inp)
    out = np.array(out)

    # First, check if the entire grid was rotated
    if (np.array_equal(rot90(inp), out) or
        np.array_equal(rot180(inp), out) or
        np.array_equal(rot270(inp), out)):
        return True

    # Then check for rotation at the object level
    if not inp_objs or not out_objs:
        return False

    # Normalize objects before comparing to make sure origin/position doesn't interfere
    def all_rotations(obj):
        return [normalize(obj),
                normalize(rot90(obj)),
                normalize(rot180(obj)),
                normalize(rot270(obj))]

    # Check if any input object matches any output object under any rotation
    for in_obj in inp_objs:
        for out_obj in out_objs:
            norm_out = normalize(out_obj)
            if any(np.array_equal(rot, norm_out) for rot in all_rotations(in_obj)):
                return True

    return False

MOVEMENT_RULES = [
    (check_translation, 1),
    (check_rotation, 1),
    (check_object_swapping, 1),
    (check_slide_to_edge, 0.9),
]
