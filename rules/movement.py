import numpy as np
from scipy.ndimage import label

def check_translation(inp, out):
    """
    Checks if the grid has been translated, meaning the position of elements shifted 
    without resizing or rotation. The grid shape remains the same, but the positions of the pixels 
    have changed.
    """
    return inp.shape == out.shape and not np.array_equal(inp, out)

def check_rotation(inp, out, angle=90):
    """
    Checks if the grid has been rotated by a specified angle (e.g., 90, 180, 270 degrees).
    The output grid will be compared against a rotated version of the input.
    """
    if angle == 90:
        return np.array_equal(np.rot90(inp), out)
    elif angle == 180:
        return np.array_equal(np.rot90(np.rot90(inp)), out)
    elif angle == 270:
        return np.array_equal(np.rot90(np.rot90(np.rot90(inp))), out)
    else:
        raise ValueError("Only 90, 180, and 270 degree rotations are supported.")
    
def check_object_translation(inp, out):
    """
    Checks if any objects have been translated (shifted in space) in the output grid.
    Identifies distinct objects using connected component labeling and compares their positions.
    """
    inp_labeled, num_labels_inp = label(inp > 0)
    out_labeled, num_labels_out = label(out > 0)

    # If the number of objects has changed, we assume a transformation.
    if num_labels_inp != num_labels_out:
        return True
    
    # Compare the positions of labeled objects
    for obj_id in range(1, num_labels_inp + 1):
        inp_coords = np.argwhere(inp_labeled == obj_id)
        out_coords = np.argwhere(out_labeled == obj_id)

        # If positions have changed significantly, consider it a translation
        if not np.array_equal(inp_coords, out_coords):
            return True
    
    return False

def check_object_rotation(inp, out, angle=90):
    """
    Checks if a specific object has been rotated by the given angle in the output grid.
    """
    inp_labeled, num_labels_inp = label(inp > 0)
    out_labeled, num_labels_out = label(out > 0)

    # Handle case where the number of objects doesn't match
    if num_labels_inp != num_labels_out:
        return False

    for obj_id in range(1, num_labels_inp + 1):
        inp_obj_mask = inp_labeled == obj_id
        out_obj_mask = out_labeled == obj_id
        
        inp_obj = inp * inp_obj_mask
        out_obj = out * out_obj_mask

        # Check rotation by comparing rotated versions of the object
        if angle == 90:
            if np.array_equal(np.rot90(inp_obj), out_obj):
                return True
        elif angle == 180:
            if np.array_equal(np.rot90(np.rot90(inp_obj)), out_obj):
                return True
        elif angle == 270:
            if np.array_equal(np.rot90(np.rot90(np.rot90(inp_obj))), out_obj):
                return True
        else:
            raise ValueError("Only 90, 180, and 270 degree rotations are supported.")
    
    return False

MOVEMENT_RULES = [
    (check_translation, 1),
    (check_rotation, 1),
    (check_object_translation, 1),
    (check_object_rotation, 1)

]
