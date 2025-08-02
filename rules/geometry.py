GEOMETRY_RULES_NL = [
    (
        "Objects stretch to edges: Detects if any input object not touching the grid edge stretches to touch the edge in the output grid.",
        1
    ),
    (
        "Object has rotated: Checks if any input object has rotated (non-symmetric shapes only) to match an output object.",
        1
    ),
    (
        "Object is completely surrounded: Determines if any input object is fully surrounded on all four sides (up, down, left, right) by output objects.",
        1
    ),
    (
        "Object removed but gap remains: Detects if an input object disappears in the output but leaves a gap of background cells of the same shape.",
        1
    ),
    (
        "Grid has rotated: Checks if the entire input grid has been rotated (by 90°, 180°, or 270°) to become the output grid.",
        1
    ),
    (
        "Object has mirrored: Detects if input objects have been mirrored (vertically, horizontally, or diagonally) in the output grid.",
        1
    )
]
