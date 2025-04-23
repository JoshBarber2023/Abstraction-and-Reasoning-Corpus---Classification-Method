import numpy as np

def check_gravity_down(inp, out):
    """
    Checks if the sum of pixel values has increased, which could suggest gravity-like 
    forces causing elements to accumulate towards the bottom (e.g., falling objects).
    """
    return inp.sum() < out.sum()

def check_containment_change(inp, out):
    """
    Verifies if the content inside the grid has changed while keeping the shape the same,
    which may indicate an interaction with the environment (e.g., an object being contained).
    """
    return inp.shape == out.shape and np.any(inp != out)

def check_ring_filling(inp, out):
    # Step 1: Identify the shapes in inp (non-zero regions, for example)
    # You can define the shape detection criteria based on your input data
    shape_mask = inp != 0  # Assume non-zero values in inp define a shape
    
    # Step 2: Initialise a grid to store filled areas
    filled_area = np.zeros_like(inp)
    
    # Step 3: Iterate through the grid to compare corresponding regions in inp and out
    # For each point in inp that defines a shape, check the corresponding region in out
    for row in range(inp.shape[0]):
        for col in range(inp.shape[1]):
            if shape_mask[row, col]:  # If this cell is part of a shape in inp
                # Define a region in out to check for filling
                # You can adjust the size of the region if necessary (example: 3x3 region around the point)
                region_size = 3
                half_region = region_size // 2
                
                # Ensure the region does not go out of bounds
                row_start = max(row - half_region, 0)
                row_end = min(row + half_region + 1, out.shape[0])
                col_start = max(col - half_region, 0)
                col_end = min(col + half_region + 1, out.shape[1])
                
                # Extract the corresponding region in out
                region_out = out[row_start:row_end, col_start:col_end]
                
                # Step 4: Check if there are any non-zero values in this region (indicating a fill)
                if np.any(region_out != 0):  # Fill is detected if non-zero values are present
                    filled_area[row, col] = 1  # Mark the position as filled
    
    return np.any(filled_area) 

COMMONSENSE_RULES = [
    (check_gravity_down, 1),
    (check_containment_change, 1),
    (check_ring_filling, 1)
]
