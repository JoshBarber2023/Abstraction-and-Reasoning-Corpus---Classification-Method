from dsl import color, palette

def check_colour_mapping(inp, out, inp_objs=None, out_objs=None):
    if not inp_objs or not out_objs:
        return len(set(v for row in inp for v in row)) != len(set(v for row in out for v in row))
    in_colors = set(color(obj) for obj in inp_objs)
    out_colors = set(color(obj) for obj in out_objs)
    return in_colors != out_colors

def check_palette_swap(inp, out, inp_objs=None, out_objs=None):
    if isinstance(inp, tuple) and isinstance(inp[0], tuple):
        in_palette = palette(inp)
        out_palette = palette(out)
    else:
        in_palette = set(v for row in inp for v in row)
        out_palette = set(v for row in out for v in row)
    return in_palette != out_palette

COLOUR_RULES = [
    (check_colour_mapping, 1),
    (check_palette_swap, 1),
]
