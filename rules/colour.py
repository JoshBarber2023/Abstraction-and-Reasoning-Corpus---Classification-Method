COLOUR_RULES_NL = [
    (
        "Colour rules describe **precise, systematic transformations of object colours or palettes** that preserve object shapes, spatial layouts, and boundaries. "
        "Typical examples include:\n"
        "- Global palette shifts where all occurrences of a specific colour are changed to another (e.g., all red pixels become blue),\n"
        "- Conditional recolouring based on object features such as size, position, or pattern (e.g., recolour only small red objects to green),\n"
        "- Hue or saturation adjustments applied consistently across objects without altering shape or location.\n"
        "Colour rules rarely alter object masks; thus, input and output object masks remain closely aligned with minimal shape deformation or positional shifts.\n"
        "Reliable detection involves:\n"
        "- Comparing dominant colours of corresponding objects,\n"
        "- Measuring colour histogram shifts,\n"
        "- Assessing whether colour mappings are one-to-one or conditionally applied,\n"
        "- Detecting minimal entropy or randomness in colour changes.\n"
        "Ambiguities arise if objects are partially recoloured, multi-coloured internally, or have slight positional jitter, but true colour rules show consistent, explainable recolouring patterns."
        , 1
    )
]
