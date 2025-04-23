### Known Bugs


2. **Rules focus on the 'entire grid', not objects within a grid:**
   - **Issue:** 
     ```
     When evaluating the rules within the different ARC input-output pairs, the rules assess the ENTIRE grid, when in reality they should focus on the 'object of interest' and evaulate from there.
     ```


### Change Log

## Added (17/04/2025):
- **JSON Evaluated Data Storing**: Implemented functionality to store evaluated data in JSON format, enabling persistent storage of processed results.
- **Progress Bar**: Integrated a progress bar to visualise task completion during long-running processes.
- **Solomonoff Score Visualisation**: Developed a bar chart to visualise Solomonoff scores for rule categories, improving data presentation and analysis.

## Added (23/04/2025):
- **Total Score Normalisation**: Added normalisation of the total scores, so they are weighted by the number of rules within the category. This was done so the number of rules doesn't 'trump' the scores.

