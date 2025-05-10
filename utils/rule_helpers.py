from dsl import * 

def to_grid(obj: Object) -> Grid:
    """Converts a set of (value, (row, col)) tuples to a minimal grid."""
    if not obj:
        return tuple()
    cells = list(obj)
    values, coords = zip(*cells)
    rows, cols = zip(*coords)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    grid = [[-1] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
    for (val, (r, c)) in cells:
        grid[r - min_r][c - min_c] = val
    return tuple(tuple(row) for row in grid)

def all_rotations(grid: Grid) -> List[Grid]:
    return [grid, rot90(grid), rot180(grid), rot270(grid)]

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