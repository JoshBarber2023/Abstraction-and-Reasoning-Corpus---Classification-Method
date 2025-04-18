### Known Bugs

1. **Category Not Displayed Under Pairs:**
   - **Issue:** The predicted category is not displayed under each pair of input/output grids when using the visualization function `compare_multiple_pairs`.
   - **Status:** In progress, a fix will be added to show predictions below the output grid.

2. **Warning for Tight Layout (Margins Issue):**
   - **Issue:** A warning message is raised when attempting to use `plt.tight_layout()`. The error message is as follows:
     ```
     c:\Users\joshl\OneDrive\Desktop\ARC Thesis\utils\visualisation.py:94: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all axes decorations.
     ```
   - **Status:** This is a known issue when using `tight_layout` with a layout that requires more space than available. A potential fix could be adjusting subplot margins or using manual layout management.

### Change Log

## Added (17/04/2025):
- **JSON Evaluated Data Storing**: Implemented functionality to store evaluated data in JSON format, enabling persistent storage of processed results.
- **Progress Bar**: Integrated a progress bar to visualize task completion during long-running processes.
- **Solomonoff Score Visualization**: Developed a bar chart to visualize Solomonoff scores for rule categories, improving data presentation and analysis.