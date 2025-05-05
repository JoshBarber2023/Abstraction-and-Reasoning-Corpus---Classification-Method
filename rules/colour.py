from dsl import color, toobject, numcolors

def check_colour_mapping(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return len(set(v for row in inp for v in row)) != len(set(v for row in out for v in row))
    in_colors = set(color(obj) for obj in inp_objs)
    out_colors = set(color(obj) for obj in out_objs)
    return in_colors != out_colors

def check_palette_swap(inp, out, inp_objs=None, out_objs=None):
    in_palette = set(v for row in inp for v in row)
    out_palette = set(v for row in out for v in row)
    return in_palette != out_palette

def check_uniform_colouring(inp, out, inp_objs=None, out_objs=None):
    """All objects in output have the same colour."""
    if not out_objs:
        return False
    out_colours = set(color(obj) for obj in out_objs)
    return len(out_colours) == 1

def check_object_colours_unique(inp, out, inp_objs=None, out_objs=None):
    """Each object has a unique colour."""
    if not out_objs:
        return False
    return len(set(color(obj) for obj in out_objs)) == len(out_objs)

def number_of_colours(inp, out, inp_objs=None, out_objs=None):
    return numcolors(inp) != numcolors(out)

COLOUR_RULES = [
    (check_colour_mapping, 1),
    (check_palette_swap, 1),
    (check_uniform_colouring, 1),
    (check_object_colours_unique, 1),
    (number_of_colours, 1)
]
