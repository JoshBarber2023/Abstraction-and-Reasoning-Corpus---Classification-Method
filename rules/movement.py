MOVEMENT_RULES_NL = [
    (
        "Movement rules describe transformations where objects or pixels are **translated spatially across the grid** without fundamentally altering shape or colour. "
        "Examples include:\n"
        "- Shifting every object by a fixed vector (e.g., all objects move right by two cells),\n"
        "- Moving specific objects to target locations such as the center or grid edges,\n"
        "- Coordinated motions where objects follow paths or track the movement of other objects.\n"
        "Detecting movement involves:\n"
        "- Matching object identities between input and output via overlap, centroid proximity, or feature similarity,\n"
        "- Calculating displacement vectors for each matched object,\n"
        "- Identifying consistent offsets across multiple objects or continuous trajectories.\n"
        "Movement may co-occur with rotations or recolouring, complicating identity tracking, so robust detection applies:\n"
        "- Cross-correlation,\n"
        "- Centroid and shape-preserving checks,\n"
        "- Distinguishing pure translation from combined geometric or colour edits."
        , 1
    )
]
