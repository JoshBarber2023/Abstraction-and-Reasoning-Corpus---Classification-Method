import numpy as np
from dsl import *

def check_filled_relationship(inp, out, inp_objs=None, out_objs=None):
    """
    Checks if, when there are more output objects than input objects, any new output object
    has appeared inside the bounds of an input object (excluding recolouring or unchanged objects).
    """
    if inp_objs is None or out_objs is None:
        return False

    # Rule only applies if number of output objects is greater than number of input objects
    if len(out_objs) <= len(inp_objs):
        return False

    # Check each output object against each input object
    for output in out_objs:
        for input_obj in inp_objs:
            # Check if the output object is inside the input object
            if all(cell in input_obj for cell in output):
                return True
    return False

def check_column_ascending_order(inp, out, inp_objs=None, out_objs=None):
    """
    Check if each column in the input grid is arranged in ascending order.
    
    Args:
        inp (Grid): Input grid.
        out (Grid): Output grid (for comparison, not used in this case).
        inp_objs (Objects, optional): Input objects (not used here).
        out_objs (Objects, optional): Output objects (not used here).
    
    Returns:
        bool: True if each column in the input grid is in ascending order, False otherwise.
    """
    # Get the number of columns (assuming the grid is non-empty and rectangular)
    num_cols = len(inp[0])
    
    # Iterate through each column
    for col in range(num_cols):
        # Check if the elements in the column are in ascending order
        for row in range(1, len(inp)):
            if inp[row][col] < inp[row - 1][col]:
                return False
    return True

def check_column_descending_order(inp, out, inp_objs=None, out_objs=None):
    """
    Check if each column in the input grid is arranged in descending order.
    
    Args:
        inp (Grid): Input grid.
        out (Grid): Output grid (for comparison, not used in this case).
        inp_objs (Objects, optional): Input objects (not used here).
        out_objs (Objects, optional): Output objects (not used here).
    
    Returns:
        bool: True if each column in the input grid is in descending order, False otherwise.
    """
    # Get the number of columns (assuming the grid is non-empty and rectangular)
    num_cols = len(inp[0])
    
    # Iterate through each column
    for col in range(num_cols):
        # Check if the elements in the column are in descending order
        for row in range(1, len(inp)):
            if inp[row][col] > inp[row - 1][col]:
                return False
    return True


COMMONSENSE_RULES = [
    (check_filled_relationship, 1),
    (check_column_ascending_order, 1),
    (check_column_descending_order, 1)
]
