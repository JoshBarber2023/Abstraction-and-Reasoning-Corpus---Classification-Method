COMMONSENSE_RULES_NL = [
    (
        "Filled relationship: Checks if the background inside or around objects was filled or cleared between input and output grids.",
        1
    ),
    (
        "Column ascending order: Verifies if objects in the output grid are aligned vertically and their values ascend from top to bottom.",
        1
    ),
    (
        "Column descending order: Verifies if objects in the output grid are aligned vertically and their values descend from top to bottom.",
        1
    ),
    (
        "Shape-to-color relationship: Checks if output object colors change based on their shapes compared to the input while maintaining some pixel overlap.",
        1
    ),
    (
        "Combined objects form common shape: Determines if combining non-overlapping input and output objects of the same color forms a common shape like square, rectangle, circle, or triangle.",
        1
    ),
    (
        "Checkerboard pattern: Checks if the output grid forms a checkerboard pattern of alternating values.",
        1
    ),
    (
        "Outputs do not overlap inputs: Ensures no output object overlaps with any input object in the grid.",
        1
    ),
    (
        "Tetris relationship: Detects Tetris-like transformations where rows are cleared, objects move downward, or gravity pulls objects to the bottom row.",
        1
    )
]
