### Known Bugs

1. **README changes:**
   - **Issue:** 
     ```
     README should describe process for running the code, and an explanation of the code and the theories behind it.
     ```

2. **Rules not working with new object detection:**
   - **Issue:** 
     ```
     Rules such as check_rotation are not currently working with the new object detection. The problem is that the rules themselves do not compare the rotation of the object and in examples where there is a clear rotation, the rule fails. Other rules need to be validated individually.
     ```

3. **Need to read through the DSL and implement it better:**
   - **Issue:** 
     ```
     The DSL is AMAZING at solving the ARC problems and provides a fundamental basis for actually solving the problems. It contains different 'patterns' which are needed for the solutions. I could reverse engineer these for my rules in order to do logical checks on whether something happened or not.
     ```


### Change Log

## Added (17/04/2025):
- **JSON Evaluated Data Storing**: Implemented functionality to store evaluated data in JSON format, enabling persistent storage of processed results.
- **Progress Bar**: Integrated a progress bar to visualise task completion during long-running processes.
- **Solomonoff Score Visualisation**: Developed a bar chart to visualise Solomonoff scores for rule categories, improving data presentation and analysis.

## Added (23/04/2025):
- **Total Score Normalisation**: Added normalisation of the total scores, so they are weighted by the number of rules within the category. This was done so the number of rules doesn't 'trump' the scores.

## Added (30/04/2025): 
- **Implemented DSL made for ARC**: As part of this DSL, object detection is now existing. The next step would be to edit the rules to properly detect the difference between the input and output object. This has been started but is not working on rules such as check_rotation which doesn't compare.



