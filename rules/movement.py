import numpy as np
from dsl import *
from rules.object import objects_get_larger, objects_get_smaller
from utils.rule_helpers import *
from rules.geometry import *



def check_object_moved(inp, out, inp_objs=None, out_objs=None):
    """
    Checks if any object has moved based on centroid position.
    Matches objects from input to output based on closest centroid.
    """
    if not inp_objs or not out_objs:
        return False
    
    if object_has_rotated(inp, out, inp_objs, out_objs) == True:
        return False
    
    if neighbour_object_disappears(inp, out, inp_objs, out_objs) == True:
        return False
    
    if shape_to_color_relationship(inp, out, inp_objs, out_objs): 
        return  False

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
    def object_to_index_map(objs: Objects) -> dict:
        """Returns a mapping from each cell to its owning object."""
        mapping = {}
        for obj in objs:
            for cell in obj:
                mapping[cell] = obj
        return mapping

    if inp_objs is None or out_objs is None:
        return False
    
    if objects_get_larger(inp, out, inp_objs, out_objs) or objects_get_smaller(inp, out, inp_objs, out_objs):
        return False

    inp_map = object_to_index_map(inp_objs)
    out_map = object_to_index_map(out_objs)

    # Check for overlap shifts: cells that change ownership
    overlap_shift_detected = False

    all_positions = set(inp_map.keys()) | set(out_map.keys())
    for pos in all_positions:
        inp_obj = inp_map.get(pos, None)
        out_obj = out_map.get(pos, None)

        if inp_obj and out_obj and inp_obj != out_obj:
            # Object ownership of this cell changed, indicating possible visual stacking change
            overlap_shift_detected = True
            break

    return overlap_shift_detected

MOVEMENT_RULES = [
    (check_object_moved, 1),
    (movement_foreground_background_shift, 1)
]
