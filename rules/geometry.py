import numpy as np
from dsl import normalize

def check_horizontal_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.flipud(out)) and np.array_equal(inp, np.flipud(inp))

def check_vertical_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.fliplr(out)) and np.array_equal(inp, np.fliplr(inp))

def check_shapes_preserved(inp, out, inp_objs=None, out_objs=None):
    inp = np.array(inp)
    out = np.array(out)

    if not inp_objs or not out_objs:
        return len(np.unique(inp[inp != 0])) == len(np.unique(out[out != 0]))
    
    return all(normalize(o1) in [normalize(o2) for o2 in out_objs] for o1 in inp_objs)

def check_shape_mirroring(inp, out, inp_objs=None, out_objs=None):
    """Input is mirrored across vertical/horizontal axis."""
    return np.array_equal(out, np.flip(inp, axis=1)) or np.array_equal(out, np.flip(inp, axis=0))

def check_diagonal_symmetry(inp, out, inp_objs=None, out_objs=None):
    return np.array_equal(out, np.fliplr(np.flipud(inp))) or np.array_equal(out, np.flipud(np.fliplr(inp)))

def check_folding_operation(inp, out, inp_objs=None, out_objs=None):
    # Assuming folding operation results in symmetry or reduced area in one direction
    return np.array_equal(out, np.flipud(np.flip(inp))) or np.array_equal(out, np.fliplr(np.flip(inp)))

def check_arbitrary_rotation(inp, out, inp_objs=None, out_objs=None, grid_size=(10, 10)):
    if inp_objs is None or out_objs is None:
        return False

    # Ensure inp_objs and out_objs are NumPy arrays
    inp_objs = np.array(inp_objs, dtype=float)
    out_objs = np.array(out_objs, dtype=float)

    # Define the 4 possible cardinal directions: 0°, 90°, 180°, 270°
    rotation_angles = [0, 90, 180, 270]
    
    # Create a list to store the rotated objects for each cardinal direction
    rotated_objs = []

    for angle in rotation_angles:
        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Create rotation matrix for 2D points
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Rotate inp_objs by the specified angle
        inp_objs_rotated = np.array([np.dot(rotation_matrix, np.array([x, y])) for x, y in inp_objs])

        # Apply modulo to wrap coordinates within grid bounds
        inp_objs_rotated = np.array([(x % grid_size[0], y % grid_size[1]) for x, y in inp_objs_rotated])

        # Append the rotated objects to the list
        rotated_objs.append(inp_objs_rotated)

    # Now check if the output matches any of the rotated versions
    # Use np.allclose to check if the output matches any rotated version
    return any(np.allclose(out_objs, rotated) for rotated in rotated_objs)

GEOMETRY_RULES = [
    (check_horizontal_symmetry, 1),
    (check_vertical_symmetry, 1),
    (check_shapes_preserved, 1),
    (check_shape_mirroring, 1),
    (check_diagonal_symmetry, 1),
    (check_folding_operation, 1),
    (check_arbitrary_rotation, 1),
]
