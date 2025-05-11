import numpy as np
from collections import Counter

def object_shape_duplicated_in_output_only(inp, out, inp_objs=None, out_objs=None):
    """
    Returns True if any object shape (irrespective of position or colour) appears
    more than once in the output, but only once or zero times in the input.
    """

    if inp_objs is None or out_objs is None:
        return False

    def get_shape(obj):
        # Normalizes object shape to origin and returns sorted relative coordinates
        min_x = min(cell[1][0] for cell in obj)
        min_y = min(cell[1][1] for cell in obj)
        return tuple(sorted((x - min_x, y - min_y) for _, (x, y) in obj))

    # Get shape representations
    inp_shapes = [get_shape(obj) for obj in inp_objs]
    out_shapes = [get_shape(obj) for obj in out_objs]

    from collections import Counter

    inp_shape_counts = Counter(inp_shapes)
    out_shape_counts = Counter(out_shapes)

    # Check if any shape appears more than once in output but at most once in input
    for shape in set(out_shapes):
        if out_shape_counts[shape] > 1 and inp_shape_counts.get(shape, 0) <= 1:
            return True

    return False



NUMBER_RULES = [
    (object_shape_duplicated_in_output_only, 1)
]
