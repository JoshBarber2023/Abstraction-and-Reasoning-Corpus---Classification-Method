from dsl import * 
import numpy as np
import math

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
    # Check if any cell in obj1 is orthogonally adjacent to any cell in obj2 (no diagonals)
    for (r1, c1) in obj1:
        for (r2, c2) in obj2:
            if (abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1):
                return True
    return False

def is_vertically_aligned(obj):
    """Check if all cells in an object are in the same column (i.e., same y)."""
    ys = [y for _, y in obj]
    return all(y == ys[0] for y in ys)

def is_horizontally_aligned(obj):
    """Check if all cells in an object are in the same row (i.e., same x)."""
    xs = [x for x, _ in obj]
    return all(x == xs[0] for x in xs)

def combine_objects(obj1, obj2):
    return obj1.union(obj2)

def is_square(obj):
    coords = [cell[1] for cell in obj]
    if not coords:
        return False
    xs, ys = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Handle the edge case for 1x1 pixel
    if (max_x - min_x == 0) and (max_y - min_y == 0):
        return False
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    expected_area = width * height
    return len(obj) == expected_area and width == height

def is_rectangle(obj):
    coords = [cell[1] for cell in obj]
    if not coords:
        return False
    xs, ys = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    # Ensure it's not a 1xN or Nx1 shape
    if width == 1 or height == 1:
        return False
    
    return len(obj) == width * height and width != height

def is_circle(obj):
    coords = [cell[1] for cell in obj]
    if not coords:
        return False
    xs, ys = zip(*coords)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)  # center of the circle
    radius = math.sqrt(sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys)) / len(xs))
    # check if all points lie within the radius (tolerance for pixel precision)
    tolerance = 2  # You can adjust the tolerance depending on your needs
    return all(abs(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius) < tolerance for x, y in zip(xs, ys))

def is_triangle(obj):
    coords = [cell[1] for cell in obj]
    if len(coords) < 3:
        return False  # A triangle needs at least 3 points
    xs, ys = zip(*coords)
    # Calculate the area of the triangle using the determinant formula
    area = 0.5 * abs(xs[0]*(ys[1] - ys[2]) + xs[1]*(ys[2] - ys[0]) + xs[2]*(ys[0] - ys[1]))
    return area > 0

def get_object_color(obj: Object) -> Union[int, None]:
    colors = {cell[0] for cell in obj}
    if len(colors) == 1:
        return next(iter(colors))
    return None  # Invalid if multiple colors present

def extract_non_empty_block(grid):
    """Extracts the minimal bounding box around non-empty cells."""
    rows = len(grid)
    cols = len(grid[0])
    top, bottom, left, right = rows, 0, cols, 0

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:  # Assuming 0 is the 'empty' value
                top = min(top, i)
                bottom = max(bottom, i)
                left = min(left, j)
                right = max(right, j)
    
    if bottom < top or right < left:
        return None  # No non-zero cells

    return [row[left:right+1] for row in grid[top:bottom+1]]

def match_pattern(grid, pattern, row, col):
    """Check if the pattern matches the grid starting at (row, col)."""
    pr, pc = len(pattern), len(pattern[0])
    for i in range(pr):
        for j in range(pc):
            if row + i >= len(grid) or col + j >= len(grid[0]):
                return False
            if grid[row + i][col + j] != pattern[i][j]:
                return False
    return True

def count_pattern_repeats(grid, pattern):
    """Count how many times the pattern fully repeats in the grid."""
    rows, cols = len(grid), len(grid[0])
    pr, pc = len(pattern), len(pattern[0])
    count = 0
    for i in range(0, rows - pr + 1):
        for j in range(0, cols - pc + 1):
            if match_pattern(grid, pattern, i, j):
                count += 1
    return count

def objects_overlap(obj1: frozenset, obj2: frozenset) -> bool:
    return not obj1.isdisjoint(obj2)

def get_bounding_box(obj):
    """Returns the bounding box of a list of (x, y) cells"""
    xs = [x for x, y in obj]
    ys = [y for x, y in obj]
    return min(xs), min(ys), max(xs), max(ys)

def get_bounding_box_area(obj):
    # obj: list of (color, location) tuples; location assumed to be (x, y)
    locations = [loc for _, loc in obj]
    if not locations:
        return 0
    xs, ys = zip(*locations)
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    return width * height

def match_objects_by_overlap(inp_objs, out_objs):
    """
    Attempts to match each input object with an output object based on location overlap.
    Returns a list of (in_obj, out_obj) pairs.
    """
    matches = []
    used_out = set()

    for in_obj in inp_objs:
        in_locs = {loc for _, loc in in_obj}
        best_match = None
        best_overlap = 0

        for i, out_obj in enumerate(out_objs):
            if i in used_out:
                continue

            out_locs = {loc for _, loc in out_obj}
            overlap = len(in_locs & out_locs)

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (in_obj, out_obj, i)

        if best_match:
            matches.append((best_match[0], best_match[1]))
            used_out.add(best_match[2])

    return matches
