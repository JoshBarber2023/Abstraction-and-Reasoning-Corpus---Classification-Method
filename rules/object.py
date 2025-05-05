import numpy as np
from dsl import *

def check_horizontal_concatenation(inp, out, inp_objs=None, out_objs=None):
    """
    Check if the output is the result of horizontally concatenating two grid parts.
    """
    for split in range(1, len(out[0])):  # Loop over possible split positions in the output grid
        left = [row[:split] for row in out]  # Left part of the split
        right = [row[split:] for row in out]  # Right part of the split

        # Check if horizontal concatenation of left and right parts forms the output grid
        if np.array_equal(np.hstack((left, right)), out) and (
            np.array_equal(inp, left) or np.array_equal(inp, right)
        ):
            return True

    return False

def check_partition(inp, out, inp_objs=None, out_objs=None):
    """
    Check if the output grid is the result of partitioning the input grid into objects based on value.
    Each object in the partition should have the same value in the grid cells.
    """
    # Partition the input grid into objects based on unique values
    partitioned_inp = partition(inp)

    # Partition the output grid similarly
    partitioned_out = partition(out)

    # Compare the partitioned grids (input vs output)
    return partitioned_inp == partitioned_out

def check_deduplication(inp, out, inp_objs=None, out_objs=None):
    """
    Checks if duplicates are removed correctly in the output grid.
    
    Args:
        inp (Grid): The input grid with objects that may contain duplicates.
        out (Grid): The output grid with objects after deduplication.
        inp_objs (List[Object]): List of input objects.
        out_objs (List[Object]): List of output objects.
    
    Returns:
        bool: True if duplicates are removed, False otherwise.
    """
    seen = set()
    for obj in out_objs:
        if obj in seen:
            return False
        seen.add(obj)
    return True

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
    (check_horizontal_concatenation, 1),
    (check_deduplication, 1)
]
