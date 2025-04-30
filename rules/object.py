from dsl import toindices

def check_new_object_created(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return sum(map(len, out)) > sum(map(len, inp))
    return len(out_objs) > len(inp_objs)

def check_object_transformed(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return not (inp == out)
    in_signatures = [sorted(toindices(obj)) for obj in inp_objs]
    out_signatures = [sorted(toindices(obj)) for obj in out_objs]
    unmatched = [sig for sig in in_signatures if sig not in out_signatures]
    return len(unmatched) > 0

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

OBJECT_RULES = [
    (check_new_object_created, 1),
    (check_object_transformed, 1),
    (check_size_scaling, 1)
]
