# Abstraction and Reasoning Corpus

## ARC Rule Category Definitions

### 1. Colour

Rules involving **changes to the colour properties** of the grid or its objects.

- **Core Ideas:** Transformation of colours without modifying objects shape or position
- **Examples:**
  - Recolouring (e.g., all red objects become blue)
  - Palette swaps (entire colour scheme transformations)
  - Colour filtering or conditionals based on colour sets

### 2. Commonsense

Rules based on **implicit reasoning**, heuristic logic, or inferred patterns not directly observable in the image.

- **Core Ideas:** Mimics intuitive or inferred reasoning
- **Examples:**
  - Completing missing patterns
  - Selecting based on common behaviour
  - Applying a rule learned from context (e.g., align outputs to a common logic)

### 3. Geometry

Rules defined by **spatial or shape-based transformations**.

- **Core Ideas:** Focus on structure, symettry, and layout of shapes
- **Examples:**
  - Detecting lines, diagonals, or grid alignments
  - Reflections, rotations, or folding operations
  - Manipulations based on geometric properties (e.g., enlosing shapes)

### 4. Movement

Rules involving **translation or repositioning** of pixels or objects.

- **Core Idea:** Spatial movement of features between input and output
- **Examples:**
  - Shifting objects by a fixed amount.
  - Following a path or aligning based on other objects.
  - Coordinated translations of all or selected elements.

### 5. Number

Rules triggered or driven by **numerical values** such as counts or arithmetic relations.

- **Core Idea:** Quantity-based logic
- **Examples:**
  - Replicating an object N times
  - Changing output based on how many of a colour exists
  - Adding or subtracting rows/columns based on object count

### 6. Object

Rules that involve **explicit object detection and manipulation**

- **Core Idea:** Interaction with entities indentified as discrete pixel groups.
- **Examples:**
  - Splitting, merging, or duplicating objects
  - Morphing shapes (e.g., squashing or stretching)
  - Changing object arrangement or identity


## Known Bugs

1. **README changes:**
   - **Issue:** 
     ```
     README should describe process for running the code, and an explanation of the code and the theories behind it.
     ```

1. **3 different input and output objects variables created in the engine:**
   - **Issue:** 
     ```
     Honestly a bit stupid, but I create three different input and output objects for visualising.. rule assessment etc. Fix that.
     ```

    

## Change Log

## Added (17/04/2025):
- **JSON Evaluated Data Storing**: Implemented functionality to store evaluated data in JSON format, enabling persistent storage of processed results.
- **Progress Bar**: Integrated a progress bar to visualise task completion during long-running processes.
- **Solomonoff Score Visualisation**: Developed a bar chart to visualise Solomonoff scores for rule categories, improving data presentation and analysis.

## Added (23/04/2025):
- **Total Score Normalisation**: Added normalisation of the total scores, so they are weighted by the number of rules within the category. This was done so the number of rules doesn't 'trump' the scores.

## Added (30/04/2025): 
- **Implemented DSL made for ARC**: As part of this DSL, object detection is now existing. The next step would be to edit the rules to properly detect the difference between the input and output object. This has been started but is not working on rules such as check_rotation which doesn't compare.
- **Improved Object Detection and Display (V2 UPDATE)**: As part of this DSL, object detection is now existing. It NOW displays them correction, but rules need to be validated. As well, a new rule has been made called check size scaling for objects which 'grow or shrink' in size. Rules such as 'check Symmetry' have been weighted to 0.5 as they are not that useful in categorising something as a lot of things are Symmetrical. 

## Added (5/05/2025):
- **Formal Defintiion of Rules**: Added a formal definition of rules into the GitHub.
- **Added heavy penalisation for excessive rule failures**: In cases where the categories are over 50% incorrect, I've added a second layer of penalities to really enforce CORRECT rules.
- **Started planning for rule implementation**: I have added some rules however I am not happy with them at all. My next goal is to go through the DSLs 'solved arc problems' file and for each 'step' in a solution, I will add a rule detecting that step. I am going to start from scratch and do that.

## Added (10/05/2025):
- **Rule Rework**: Started from scratch with the rules and slowly built them through fundamental ideas of what could happen in a puzzle. I've minimised the amount of rules which could be 'too broad' and kept them to the ideas that a child would fundamentally have... "I've seen something move..." or "I can see something being arrange from smallest to biggest.." or "I can see the object getting smaller or larger."

## Added (11/05/2025):
- **Rule Rework**: Added new rules and verified each rule working in each implementation.

## Added (12/05/2025):
- **Rule Rework**: Added new rules and verified each rule working in each implementation.

## Added (12/05/2025):
- **Rule Rework**: Adjusted and verified each rule working in unit tests.
- **Evaluation Tests**: Made function to manually categorise evaluation set, then ran the solver on to compare how successful it was.