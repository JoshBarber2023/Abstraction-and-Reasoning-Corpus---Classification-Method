import numpy as np
from dsl import *
from rules.object import objects_get_larger, objects_get_smaller
from utils.rule_helpers import *
from rules.object import *



def check_object_moved(inp, out, inp_objs=None, out_objs=None):
    """
    Checks if any object has moved based on centroid position.
    Matches objects from input to output based on closest centroid.
    """
    if not inp_objs or not out_objs:
        return False
    
    if object_has_rotated(inp, out, inp_objs, out_objs):
        return False
    
    if neighbour_object_disappears(inp, out, inp_objs, out_objs):
        return False
    
    if shape_to_color_relationship(inp, out, inp_objs, out_objs): 
        return  False
    
    if objects_get_smaller(inp, out, inp_objs, out_objs) or objects_get_larger(inp, out, inp_objs, out_objs):
        return False

    # Compute centroids for all objects
    inp_centroids = [get_centroid(obj) for obj in inp_objs]
    out_centroids = [get_centroid(obj) for obj in out_objs]

    # Track movement: for each input object, check if any output object is nearby
    for inp_c in inp_centroids:
        # Find closest output centroid
        distances = [(out_c, ((inp_c[0] - out_c[0]) ** 2 + (inp_c[1] - out_c[1]) ** 2) ** 0.5)
                     for out_c in out_centroids]
        closest_out_c, dist = min(distances, key=lambda x: x[1])

        # Consider as moved if the centroid changed by more than a small tolerance
        if dist > 1e-2:
            return True

    return False

def movement_foreground_background_shift(inp, out, inp_objs=None, out_objs=None) -> bool:
    def object_to_index_map(objs):
        """Returns a mapping from each cell to its owning object."""
        mapping = {}
        for obj in objs:
            for cell in obj:
                mapping[cell] = obj
        return mapping

    if inp_objs is None or out_objs is None:
        return False

    # Skip if it's just a growth/shrink operation
    if objects_get_larger(inp, out, inp_objs, out_objs) or objects_get_smaller(inp, out, inp_objs, out_objs):
        return False

    inp_map = object_to_index_map(inp_objs)
    out_map = object_to_index_map(out_objs)

    # Focus only on overlapping cells — those shared by both input and output
    overlapping_positions = set(inp_map.keys()) & set(out_map.keys())

    for pos in overlapping_positions:
        inp_obj = inp_map[pos]
        out_obj = out_map[pos]
        if inp_obj != out_obj:
            # True foreground/background reassignment
            return True

    return False

def objects_fall_downward(inp, out, inp_objs=None, out_objs=None):
    """
    Checks if all objects have moved downward in the grid (gravity rule).
    """
    if not inp_objs or not out_objs:
        return False

    # Early out if objects are rotating or changing size
    if object_has_rotated(inp, out, inp_objs, out_objs):
        return False
    if objects_get_larger(inp, out, inp_objs, out_objs) or objects_get_smaller(inp, out, inp_objs, out_objs):
        return False

    # Match input objects to output objects based on centroid proximity
    inp_centroids = [get_centroid(obj) for obj in inp_objs]
    out_centroids = [get_centroid(obj) for obj in out_objs]

    matched = [False] * len(out_objs)

    for inp_idx, inp_c in enumerate(inp_centroids):
        best_match = None
        min_dist = float('inf')
        best_idx = None

        for out_idx, out_c in enumerate(out_centroids):
            if matched[out_idx]:
                continue
            dist = ((inp_c[0] - out_c[0]) ** 2 + (inp_c[1] - out_c[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_match = out_c
                best_idx = out_idx

        if best_match is None:
            return False  # Couldn't find a match

        matched[best_idx] = True

        # Gravity should pull objects downward (i.e., y should increase)
        if best_match[0] < inp_c[0]:  # Row coordinate decreased = moved up
            return False

    return True

MOVEMENT_RULES = [
    (check_object_moved, 1),
    (movement_foreground_background_shift, 1),
    (objects_fall_downward, 1)
]