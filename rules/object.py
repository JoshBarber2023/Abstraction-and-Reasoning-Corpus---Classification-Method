OBJECT_RULES_NL = [
    (
        "Object rules treat connected pixel groups as **discrete entities** and apply **structural transformations** on these entities. "
        "Typical operations include:\n"
        "- Splitting a single object into multiple parts,\n"
        "- Merging several objects into one unified shape,\n"
        "- Resizing or stretching objects,\n"
        "- Altering topology such as filling holes, adding protrusions, or changing connectivity,\n"
        "- Reassigning object identities or order based on structural changes.\n"
        "These tasks require:\n"
        "- Robust connected-component labelling (including diagonal connectivity options),\n"
        "- Graph-based analyses capturing adjacency and containment relations,\n"
        "- Identity-tracking methods capable of mapping input objects to output objects despite topological changes.\n"
        "Unlike pure geometry or movement rules, object rules explicitly modify the **entity structure** by creating, destroying, or morphing objects, not just repositioning or rotating them."
        , 1
    )
]
