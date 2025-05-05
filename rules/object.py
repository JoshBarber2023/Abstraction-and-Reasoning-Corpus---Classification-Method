import numpy as np
from dsl import normalize, toindices, centerofmass

def check_object_duplication(inp, out, inp_objs=None, out_objs=None):
    """Objects are duplicated if their count increases, and they match the pattern of input objects."""
    return len(out_objs) > len(inp_objs) and any(normalize(out_obj) != normalize(inp_obj) for out_obj in out_objs for inp_obj in inp_objs)

def check_object_merging(inp, out, inp_objs=None, out_objs=None):
    """Objects merge if the count of output objects decreases."""
    return len(out_objs) < len(inp_objs) and any(normalize(out_obj) != normalize(inp_obj) for out_obj in out_objs for inp_obj in inp_objs)

def check_object_alignment(inp, out, inp_objs=None, out_objs=None):
    """Check if objects are aligned based on center of mass."""
    if not inp_objs or not out_objs:
        return False
    centers_in = [centerofmass(obj) for obj in inp_objs]
    centers_out = [centerofmass(obj) for obj in out_objs]
    
    tolerance = 2  # Adjust tolerance as needed
    return all(abs(c[0] - centers_out[0][0]) < tolerance for c in centers_in) or all(abs(c[1] - centers_out[0][1]) < tolerance for c in centers_in)

def check_new_object_created(inp, out, inp_objs=None, out_objs=None):
    """Check if new objects are created (more objects in output)."""
    return len(out_objs) > len(inp_objs)

def check_object_transformed(inp, out, inp_objs=None, out_objs=None):
    """Objects are transformed if their structure has changed (e.g., different set of indices)."""
    if not inp_objs or not out_objs:
        return inp != out
    in_signatures = [sorted(toindices(obj)) for obj in inp_objs]
    out_signatures = [sorted(toindices(obj)) for obj in out_objs]
    unmatched = [sig for sig in in_signatures if sig not in out_signatures]
    return len(unmatched) > 0

OBJECT_RULES = [
    (check_object_duplication, 1),
    (check_object_merging, 1),
    (check_object_alignment, 1),
    (check_new_object_created, 1),
    (check_object_transformed, 1),
]
