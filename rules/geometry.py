import numpy as np
from dsl import *
from utils.rule_helpers import *
from rules.commonsense import *

def objects_stretch_to_edges(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False
    
    out_shape = (len(out), len(out[0]))

    for in_obj in inp_objs:
        best_match = None
        max_overlap = 0
        for out_obj in out_objs:
            overlap = len(in_obj & out_obj)
            if overlap > max_overlap:
                best_match = out_obj
                max_overlap = overlap

        if best_match is None:
            continue

        if not touches_edge(in_obj, out_shape) and touches_edge(best_match, out_shape):
            return True

    return False

def object_has_rotated(inp, out, inp_objs=None, out_objs=None):
    from itertools import product

    if not inp_objs or not out_objs:
        return False
    
    if check_filled_relationship(inp, out, inp_objs, out_objs):
        return False

    inp_grids = [to_grid(obj) for obj in inp_objs]
    out_grids = [to_grid(obj) for obj in out_objs]

    def is_rotationally_symmetric(grid):
        # Check if grid is the same under all 90-degree rotations
        return all(rot == grid for rot in all_rotations(grid))

    for inp_grid in inp_grids:
        # Skip objects that can't visually rotate
        if sum(cell is not None for row in inp_grid for cell in row) == 1:
            continue  # single pixel
        if is_rotationally_symmetric(inp_grid):
            continue  # e.g., square symmetric object

        matched = False
        for out_grid in out_grids:
            if any(rot == out_grid for rot in all_rotations(inp_grid)):
                matched = True
                break
        if not matched:
            return True

    return False

def is_completely_surrounded(inp, out, inp_objs=None, out_objs=None):
    if not out_objs or not inp_objs:
        return False

    # Define directions: up, down, left, right (4-connectivity)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for inp_obj in inp_objs:
        # For each input object, we need to check all its edge cells
        all_sides_covered = True
        for cell in inp_obj:
            x, y = cell
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                # If neighbor is not in inp_obj and is also not in any out_obj, it's not surrounded
                if neighbor not in inp_obj and not any(neighbor in out_obj for out_obj in out_objs):
                    all_sides_covered = False
                    break
            if not all_sides_covered:
                break

        if all_sides_covered:
            return True  # At least one input object is fully surrounded

    return False

def object_removed_but_gap_remains(inp, out, inp_objs=None, out_objs=None):
    if inp_objs is None or out_objs is None:
        return False

    # Create a set of all output cells
    out_cells = set()
    for obj in out_objs:
        out_cells.update(obj)

    for obj in inp_objs:
        # Check if object is completely gone from output
        if all(cell not in out_cells for cell in obj):
            # Check if the same shape is left as empty in the output
            shape_cells = {(r, c) for (val, (r, c)) in obj}
            values_at_positions = [out[r][c] for (r, c) in shape_cells]

            # If all those values are 0 or a background colour (assumed 0), then it's a "gap"
            if all(v == 0 for v in values_at_positions):
                return True

    return False

def check_surrounded_cell_disappearance(inp, out, inp_objs=None, out_objs=None):
    # Iterate through all cells in the input objects
    for obj in inp_objs:
        for cell in obj:
            # Get the position of the cell and check if it's surrounded by other cells in the input grid
            x, y = cell[1]
            
            # Check the 8 neighboring positions (all directions) around the cell
            neighbors = [
                (x-1, y-1), (x, y-1), (x+1, y-1),
                (x-1, y),               (x+1, y),
                (x-1, y+1), (x, y+1), (x+1, y+1)
            ]
            
            # Check if the neighboring cells are all part of the input object
            surrounded = all((neighbor in inp for neighbor in neighbors))
            
            if surrounded:
                # Check if the cell no longer exists in the output grid
                if cell not in out:
                    return True  # The cell disappeared as expected
                
    return False  # No disappearance detected

def grid_has_rotated(inp, out, inp_objs=None, out_objs=None):
    def rotate_90(grid):
        return [list(row) for row in zip(*grid[::-1])]

    def all_rotations(grid):
        rot_90 = rotate_90(grid)
        rot_180 = rotate_90(rot_90)
        rot_270 = rotate_90(rot_180)
        return [rot_90, rot_180, rot_270]

    # Ensure dimensions are compatible for rotation
    if len(inp) != len(out[0]) or len(inp[0]) != len(out):
        return False

    for rotated in all_rotations(inp):
        if np.array_equal(rotated, out):
            return True

    return False

def object_has_mirrored(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return False

    inp_grids = [to_grid(obj) for obj in inp_objs]
    out_grids = [to_grid(obj) for obj in out_objs]

    def all_mirrors(grid):
        # Generates all mirrored versions of a grid
        vertical = [row[::-1] for row in grid]
        horizontal = grid[::-1]
        diagonal_main = [list(row) for row in zip(*grid)]  # transpose
        diagonal_anti = [list(row[::-1]) for row in zip(*grid[::-1])]  # flip then transpose
        return [vertical, horizontal, diagonal_main, diagonal_anti]

    for inp_grid in inp_grids:
        matched = False
        for out_grid in out_grids:
            if any(np.array_equal(mirror, out_grid) for mirror in all_mirrors(inp_grid)):
                matched = True
                break
        if not matched:
            return False

    return True



GEOMETRY_RULES = [
    (objects_stretch_to_edges, 1),
    (object_has_rotated, 1),
    (is_completely_surrounded, 1),
    (object_removed_but_gap_remains, 1),
    (grid_has_rotated, 1),
    (object_has_mirrored, 1),
]