from dsl import *
import numpy as np

def check_colour_substitution(inp, out, inp_objs=None, out_objs=None):
    """
    Check if the output is the result of substituting one colour value with another.
    """
    inp = np.array(inp)
    out = np.array(out)

    unique_inp = set(np.unique(inp))
    unique_out = set(np.unique(out))

    if len(unique_inp) == len(unique_out):
        # Might be a palette swap — not a single substitution
        return False

    # Look for a single substitution
    for color_in in unique_inp:
        for color_out in unique_out:
            if color_out not in unique_inp:
                # Replace candidate
                test = np.where(inp == color_in, color_out, inp)
                if np.array_equal(test, out):
                    return True

    return False

def check_colour_switch(inp, out, inp_objs=None, out_objs=None):
    """
    Check if the output is the result of swapping two colours in the grid.
    """
    inp = np.array(inp)
    out = np.array(out)

    unique_inp = set(np.unique(inp))
    unique_out = set(np.unique(out))

    if len(unique_inp) != len(unique_out):
        return False

    # Try all possible color pairs to check for a valid switch
    for color_in1 in unique_inp:
        for color_in2 in unique_inp:
            if color_in1 == color_in2:
                continue
            temp = inp.copy()
            temp = np.where(temp == color_in1, -1, temp)
            temp = np.where(temp == color_in2, color_in1, temp)
            temp = np.where(temp == -1, color_in2, temp)
            if np.array_equal(temp, out):
                return True

    return False


def check_most_common_colour(inp, out, inp_objs=None, out_objs=None):
    """
    Rule that checks if the most common color in the input grid matches the output.
    """
    inp = np.array(inp)
    out = np.array(out)

    inp_values = inp.flatten()
    out_values = out.flatten()

    most_common_inp = np.bincount(inp_values).argmax()
    most_common_out = np.bincount(out_values).argmax()

    return most_common_inp == most_common_out


def check_colour_mapping(inp, out, inp_objs=None, out_objs=None):
    """
    Check if the set of colours used in objects has changed from input to output.
    """
    if not inp_objs or not out_objs:
        return False
    in_colors = set(color(obj) for obj in inp_objs)
    out_colors = set(color(obj) for obj in out_objs)
    return in_colors != out_colors

def check_palette_swap(inp, out, inp_objs=None, out_objs=None):
    """
    Check if the entire palette has changed (not just objects).
    """
    in_palette = set(v for row in inp for v in row)
    out_palette = set(v for row in out for v in row)
    return in_palette != out_palette

def check_uniform_colouring(inp, out, inp_objs=None, out_objs=None):
    """
    All objects in output have the same colour.
    """
    if not out_objs:
        return False
    out_colours = set(color(obj) for obj in out_objs)
    return len(out_colours) == 1

def check_object_colours_unique(inp, out, inp_objs=None, out_objs=None):
    """
    Each object has a unique colour.
    """
    if not out_objs:
        return False
    return len(set(color(obj) for obj in out_objs)) == len(out_objs)

def check_number_of_colours(inp, out, inp_objs=None, out_objs=None):
    """
    Number of distinct colours changed.
    """
    return numcolors(inp) != numcolors(out)

COLOUR_RULES = [
    (check_colour_substitution, 1),
    (check_colour_switch, 1),
    (check_most_common_colour, 1),
    (check_colour_mapping, 1),
    (check_palette_swap, 1),
    (check_uniform_colouring, 1),
    (check_object_colours_unique, 1),
    (check_number_of_colours, 1),
]
