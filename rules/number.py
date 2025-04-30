import numpy as np

def check_pixel_count_change(inp, out, inp_objs=None, out_objs=None):
    return np.count_nonzero(inp) != np.count_nonzero(out)

def check_object_count_change(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return len(np.unique(inp)) != len(np.unique(out))
    return len(inp_objs) != len(out_objs)

def check_size_scaling(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs or len(inp_objs) != len(out_objs):
        return False

    # Precompute sizes (avoids building index sets)
    in_sizes = [sum(1 for _ in obj) for obj in inp_objs]
    out_sizes = [sum(1 for _ in obj) for obj in out_objs]

    try:
        ratios = set((out * 1.0) / inp for inp, out in zip(in_sizes, out_sizes) if inp != 0)
    except ZeroDivisionError:
        return False

    return len(ratios) == 1  # All scale factors must be the same

NUMBER_RULES = [
    (check_pixel_count_change, 1),
    (check_object_count_change, 1),
    (check_size_scaling, 1)
]
