from dsl import * 
import numpy as np

def normalise_object(obj):
    """Translate object cells so the top-left cell is at (0, 0)."""
    positions = [cell[1] for cell in obj]
    min_x = min(x for x, y in positions)
    min_y = min(y for x, y in positions)
    return [(val, (x - min_x, y - min_y)) for val, (x, y) in obj]

def to_grid(obj):
    """Convert object to a grid (2D list) from normalized cells."""
    norm_obj = normalise_object(obj)
    positions = [pos for _, pos in norm_obj]
    max_x = max(x for x, y in positions)
    max_y = max(y for x, y in positions)
    grid = [[None for _ in range(max_y + 1)] for _ in range(max_x + 1)]
    for val, (x, y) in norm_obj:
        grid[x][y] = val
    return grid

def all_rotations(grid: Grid) -> List[Grid]:
    return [rot90(grid), rot180(grid), rot270(grid)]

def touches_edge(obj, shape):
    rows, cols = shape
    for _, (r, c) in obj:
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            return True
    return False

def object_inside(inp_obj, out_obj):
    """Returns True if inp_obj is completely inside out_obj."""
    # Check if every cell in inp_obj is inside out_obj
    for cell in inp_obj:
        if not any(cell[1] == out_cell[1] for out_cell in out_obj):
            return False
    return True

def match_objects_by_shape_and_position(inp_objs: Objects, out_objs: Objects):
    """
    Attempt to match input and output objects based on shape (relative pixel structure)
    and approximate location (center of mass).
    """
    matched = []
    used_out_objs = set()

    for in_obj in inp_objs:
        in_shape = {cell[1] for cell in in_obj}
        in_com = centerofmass(in_obj)
        best_match = None
        best_distance = float('inf')

        for out_obj in out_objs:
            if out_obj in used_out_objs:
                continue

            out_shape = {cell[1] for cell in out_obj}
            out_com = centerofmass(out_obj)

            if in_shape == out_shape:
                # Prefer shape match; use COM distance to refine
                distance = np.linalg.norm(np.array(out_com) - np.array(in_com))
                if distance < best_distance:
                    best_match = out_obj
                    best_distance = distance

        if best_match:
            matched.append((in_obj, best_match))
            used_out_objs.add(best_match)

    return matched

def get_centroid(obj):
    """Returns the centroid of a list of (value, (x, y)) cells."""
    xs = [cell[1][0] for cell in obj]
    ys = [cell[1][1] for cell in obj]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def objects_are_neighbours(obj1, obj2):
    # Check if any cell in obj1 is adjacent (including diagonals) to any cell in obj2
    for (r1, c1) in obj1:
        for (r2, c2) in obj2:
            if abs(r1 - r2) <= 1 and abs(c1[0] - c2[0]) <= 1:
                return True
    return False
