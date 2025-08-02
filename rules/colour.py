COLOUR_RULES_NL = [
    (
        "All objects change their colour: "
        "Every object present in the input grid changes to a different colour in the output grid, "
        "while maintaining their original shape and position.",
        1
    ),
    (
        "Mimic colour scheme: "
        "If there are more than two distinct colours across the combined input and output grids, "
        "the primary colour of each object in the input is preserved in the corresponding object in the output, "
        "mirroring the overall colour palette.",
        1
    ),
    (
        "Partial internal colour change: "
        "When an object in the output is a smaller version of an input object and a neighbouring object appears, "
        "at least part of the object's internal colouring changes compared to the original.",
        1
    )
]
