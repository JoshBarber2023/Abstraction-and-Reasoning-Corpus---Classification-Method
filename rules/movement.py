MOVEMENT_RULES_NL = [
    (
        "Object moved: Detects if any object has changed position based on centroid movement, ignoring rotations, resizing, and neighbour disappearance.",
        1
    ),
    (
        "Foreground-background shift: Checks if any cells have switched their object membership between input and output, indicating a foreground/background reassignment.",
        1
    ),
    (
        "Objects fall downward: Detects if all objects have moved downward in the grid (gravity effect), without rotation or size changes.",
        1
    )
]
