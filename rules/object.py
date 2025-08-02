OBJECT_RULES_NL = [
    (
        "Objects get larger: Detects if any object in the output is larger than its corresponding input object.",
        1
    ),
    (
        "Objects get smaller: Detects if any object in the output is smaller than its corresponding input object.",
        1
    ),
    (
        "Neighbour object disappears: Detects if an object disappears in the output while one of its neighbours remains in the input.",
        1
    ),
    (
        "Neighbour object appears: Detects if a new object appears in the output adjacent to an existing object.",
        1
    ),
    (
        "Object duplication: Checks if any input object is directly duplicated and repeated multiple times in the output grid.",
        1
    )
]

