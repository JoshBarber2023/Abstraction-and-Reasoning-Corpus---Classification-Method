### Known Bugs


2. **Rules focus on the 'entire grid', not objects within a grid:**
   - **Issue:** 
     ```
     When evaluating the rules within the different ARC input-output pairs, the rules assess the ENTIRE grid, when in reality they should focus on the 'object of interest' and evaulate from there.
     ```


### Change Log

## Added (17/04/2025):
- **JSON Evaluated Data Storing**: Implemented functionality to store evaluated data in JSON format, enabling persistent storage of processed results.
- **Progress Bar**: Integrated a progress bar to visualize task completion during long-running processes.
- **Solomonoff Score Visualization**: Developed a bar chart to visualize Solomonoff scores for rule categories, improving data presentation and analysis.