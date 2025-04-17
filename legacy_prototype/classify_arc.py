import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label

# === OUTPUT EXPLANATION ===

# Each category is evaluated with multiple rules that check for certain conditions or transformations in the task's data.
# For each rule, the following details are printed in terminal:
# 
# - **Rule Name**: The rule applied to the task to check for a specific condition (e.g., check_colour_mapping).
# - **Passed all pairs**: Indicates whether the rule passed for all data pairs in the task. If True, all pairs met the condition; if False, at least one pair did not.
# - **Complexity**: The computational complexity of the rule. Lower complexity indicates a simpler rule, while higher complexity indicates a more involved rule.
# - **Prior**: The prior probability of the rule's validity, showing how likely the rule is to be correct before any data is evaluated.
# - **Passed per pair**: A list of boolean values indicating whether the rule passed for each individual data pair.
# 
# For each category, the total **Solomonoff Score** is calculated by combining the results of all rules in that category. The score is influenced by:
# 
# 1. The **complexity** of each rule (simpler rules are preferred).
# 2. The **prior probability** of the rule (how likely it is to be correct before observing data).
# 3. The **rule results** (whether the rule passed or failed for each data pair).
# 
# The Solomonoff Score is calculated as a product of the prior probability of the rule and the complexity of the rule, adjusted by the rule's performance (whether it passed or failed for each pair). Rules with higher complexity or lower prior probabilities have less impact on the final score. If a rule fails a pair, its contribution to the score is reduced.
# 
# For example:
# - In the **Object** category:
#   - The rule **check_new_object_created** passed all pairs with a high prior probability and low complexity, contributing positively to the score.
#   - The rule **check_object_transformed** failed all pairs with a lower prior probability and higher complexity, which negatively impacted the score.
# 
# The **Total Solomonoff Score** for each category is the sum of the individual scores from all rules in that category. In this case, the **Object** category has a score of 2.52435e-29, meaning that the rules in this category, although they had some positive contribution, did not provide strong evidence for a correct prediction compared to other categories.
#
# After evaluating all categories, the output predicts the category that best fits the task based on the highest Solomonoff score.

# === LINKS ===
# Solomonoff: https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference
# NOT AS FORMAL but still interesting: https://www.lesswrong.com/posts/Kyc5dFDzBg4WccrbK/an-intuitive-explanation-of-solomonoff-induction
# Page 5: https://arxiv.org/pdf/1111.6117
# How categories were chosen (Although, this needs a bit of work... I've looked at the 16 identified which are more specific and less broad..): https://openreview.net/pdf?id=F4RNpByoqP 

# === Rule Set ===
# The task uses different rules to check how input and output grids change.
# Simple rules work for many cases, but complex transformations might need more detailed rules.
# It's important to choose the right level of detail for the rules. Too specific rules don't work well,
# and rules should focus on basic concepts like whether something rotates or moves.

# === Limitations and Future Improvements ===
# There are issues with the current approach:
# - It doesn't recognise specific objects, only overall changes in the input-output grids.
# - Right now, this is just a proof of concept. The model is too simple and needs improvements to classify ARC tasks reliably.

# Future improvements should focus on:
# 1. **Object recognition:** Identifying individual objects in the grids, not just overall changes.
# 2. **Handling complex transformations:** Making sure the system handles complicated transformations like ring filling correctly.
# 3. **Better rules:** Expanding the rules to handle more transformation types.
# 4. **Balancing simplicity and complexity:** Ensuring the system can handle both simple and complex cases effectively.

# These improvements will help the system work well for the full range of ARC tasks.

# THIS CODE IMPLEMENTATION WAS A VERY LAST MINUTE, CHATGPT'd CODE FOR A PROOF OF CONCEPT... I WAS USING IT TO TEST AND VISUALISE HOW THIS COULD WORK...

# TODO: Ideally, I will write this code completely from scratch and think better into what these 'rules' could be. However, I just wanted to throw together a little prototype to visualise this very complicated, currently not-touched methodology... 
# I could not see this implementation anywhere from my google searches... I find it very interesting.


#TODO: When running this code go to __main__ and feel free to adjust between:
#     arc_folder = "./MINI-ARC/data/MiniARC" # MiniARC
#     arc_folder = "./Training" # Standard ARC

# === CATEGORY DEFINITIONS ===
CATEGORIES = ["Colour", "CommonSense", "Geometry", "Movement", "Number", "Object"]

# === Rule complexity function ===
def rule_complexity(rule_func):
    """
    A rough proxy for rule complexity using bytecode length.
    This function returns the length of the bytecode of the given rule function.
    A larger bytecode length typically corresponds to a more complex rule.
    """
    return len(rule_func.__code__.co_code)

# === Colour Rules ===

def check_colour_mapping(inp, out):
    """
    Checks if the number of unique colours has changed between input and output.
    This rule verifies if colour mapping has altered the distinct colour palette.
    """
    return np.unique(inp).size != np.unique(out).size

def check_palette_swap(inp, out):
    """
    Verifies if the order of unique colours in the input and output has been swapped.
    This rule identifies colour palette swaps by checking for differences in colour order.
    """
    return sorted(np.unique(inp)) != sorted(np.unique(out))

# === CommonSense Rules ===

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
    
    # Step 2: Initialize a grid to store filled areas
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

# === Geometry Rules ===

def check_horizontal_symmetry(inp, out):
    """
    Checks if the input and output grids exhibit horizontal symmetry (mirror reflection).
    """
    return np.array_equal(inp, np.flip(out, axis=1))

# === Geometry Rules ===

def check_horizontal_symmetry(inp, out):
    """
    Checks if the input and output grids exhibit horizontal symmetry (mirror reflection).
    """
    return np.array_equal(inp, np.flip(out, axis=1))

def check_vertical_symmetry(inp, out):
    """
    Checks if the input and output grids exhibit vertical symmetry (mirror reflection).
    """
    return np.array_equal(inp, np.flip(out, axis=0))

# === Movement Rules ===

def check_translation(inp, out):
    """
    Checks if the grid has been translated, meaning the position of elements shifted 
    without resizing or rotation. The grid shape remains the same, but the positions of the pixels 
    have changed.
    """
    return inp.shape == out.shape and not np.array_equal(inp, out)

def check_rotation(inp, out, angle=90):
    """
    Checks if the grid has been rotated by a specified angle (e.g., 90, 180, 270 degrees).
    The output grid will be compared against a rotated version of the input.
    """
    if angle == 90:
        return np.array_equal(np.rot90(inp), out)
    elif angle == 180:
        return np.array_equal(np.rot90(np.rot90(inp)), out)
    elif angle == 270:
        return np.array_equal(np.rot90(np.rot90(np.rot90(inp))), out)
    else:
        raise ValueError("Only 90, 180, and 270 degree rotations are supported.")
    
def check_object_translation(inp, out):
    """
    Checks if any objects have been translated (shifted in space) in the output grid.
    Identifies distinct objects using connected component labeling and compares their positions.
    """
    inp_labeled, num_labels_inp = label(inp > 0)
    out_labeled, num_labels_out = label(out > 0)

    # If the number of objects has changed, we assume a transformation.
    if num_labels_inp != num_labels_out:
        return True
    
    # Compare the positions of labeled objects
    for obj_id in range(1, num_labels_inp + 1):
        inp_coords = np.argwhere(inp_labeled == obj_id)
        out_coords = np.argwhere(out_labeled == obj_id)

        # If positions have changed significantly, consider it a translation
        if not np.array_equal(inp_coords, out_coords):
            return True
    
    return False

def check_object_rotation(inp, out, angle=90):
    """
    Checks if a specific object has been rotated by the given angle in the output grid.
    """
    inp_labeled, num_labels_inp = label(inp > 0)
    out_labeled, num_labels_out = label(out > 0)

    # Handle case where the number of objects doesn't match
    if num_labels_inp != num_labels_out:
        return False

    for obj_id in range(1, num_labels_inp + 1):
        inp_obj_mask = inp_labeled == obj_id
        out_obj_mask = out_labeled == obj_id
        
        inp_obj = inp * inp_obj_mask
        out_obj = out * out_obj_mask

        # Check rotation by comparing rotated versions of the object
        if angle == 90:
            if np.array_equal(np.rot90(inp_obj), out_obj):
                return True
        elif angle == 180:
            if np.array_equal(np.rot90(np.rot90(inp_obj)), out_obj):
                return True
        elif angle == 270:
            if np.array_equal(np.rot90(np.rot90(np.rot90(inp_obj))), out_obj):
                return True
        else:
            raise ValueError("Only 90, 180, and 270 degree rotations are supported.")
    
    return False

# === Number Rules ===

def check_pixel_count_change(inp, out):
    """
    Checks if the number of non-zero pixels has changed between the input and output.
    This could indicate a transformation that adds or removes pixels.
    """
    return np.count_nonzero(inp) != np.count_nonzero(out)

def check_object_count_change(inp, out):
    """
    Checks if the number of unique objects (groups of adjacent pixels) has changed.
    This may suggest that objects were created, merged, or deleted.
    """
    return np.unique(inp).size != np.unique(out).size

# === Object Rules ===

def check_new_object_created(inp, out):
    """
    Checks if a new object has been created in the output grid.
    This rule identifies the creation of new non-zero elements.
    """
    return not np.array_equal(inp, out) and np.any(out > 0)

def check_object_transformed(inp, out):
    """
    Checks if an object has been transformed (its pixel content changed) but the overall structure remains.
    The sorted content of the object remains the same, possibly rearranged.
    """
    return np.array_equal(np.sort(inp, axis=None), np.sort(out, axis=None)) and not np.array_equal(inp, out)

# === RULE SETS ===
CATEGORY_RULES = {
    "Colour": [check_colour_mapping, check_palette_swap],  # Basic colour consistency and palette changes
    "CommonSense": [check_gravity_down, check_containment_change],  # Gravity should pull down and rings should fill up
    "Geometry": [check_horizontal_symmetry, check_vertical_symmetry],  # Objects should be symmetrically aligned both ways
    "Movement": [check_translation, check_rotation],  # Objects should be able to move and rotate
    "Number": [check_pixel_count_change, check_object_count_change],  # Pixel and object count should be consistent
    "Object": [check_new_object_created, check_object_transformed],  # New objects can appear and change
}


# === Solomonoff-style scoring function with detailed prints ===
def solomonoff_score(rules, train_pairs, category_name=None):
    """
    Evaluates the total Solomonoff score for a category based on its rules.
    The score is based on the complexity of the rules and their ability to classify the training pairs.
    """
    total_score = 0.0
    num_rules = len(rules)  # Get the number of rules in the category

    for rule in rules:
        rule_name = rule.__name__
        passed_pairs = [rule(inp, out) for inp, out in train_pairs]
        success = all(passed_pairs)

        complexity = rule_complexity(rule)
        prior = 2 ** (-complexity) #PUNISHED BY OCCAMS RAZOR

        # TODO: INVESTIGATE HOW THIS SCALE CHANGES ACCURACY.. 
        #complexity_scale = 50  # Tunable: try 50–100... Needed as some complex functions (ring function) gets ignored due to its Solomonoff being close to 0 (because its a computationally heavy function)
        #prior = 2 ** (-complexity / complexity_scale)

        if success:
            total_score += prior

        print(f"    Rule: {rule_name}")
        print(f"      Passed all pairs: {success}")
        print(f"      Complexity: {complexity}, Prior: {prior:.5e}")
        print(f"      Passed per pair: {passed_pairs}")

    # Normalize the total score by the number of rules in this category
    normalized_score = total_score / num_rules

    return normalized_score


# === Visualise all input-output pairs with prediction ===
def visualise_all_train_pairs_with_prediction(task_data, task_name, predicted_category):
    """
    Visualises the input-output pairs along with the predicted category.
    This function generates a grid of input-output pairs with predictions for easy inspection.
    """
    train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task_data['train']]

    # Number of pairs
    num_pairs = len(train_pairs)

    # Create a figure with enough subplots to display input-output pairs in columns
    fig, axs = plt.subplots(3, num_pairs, figsize=(8 * num_pairs, 12))  # 3 rows: input, output, prediction
    fig.suptitle(f'{task_name} - All Input-Output Pairs with Prediction', fontsize=16)

    # If there is only one pair, axs won't be a 2D array
    if num_pairs == 1:
        axs = np.expand_dims(axs, axis=1)

    for idx, (inp, out) in enumerate(train_pairs):
        # Show the input image on the top row of the current column
        axs[0, idx].imshow(inp, cmap='viridis', interpolation='nearest')
        axs[0, idx].set_title(f"Input {idx+1}")
        axs[0, idx].axis('off')

        # Show the output image on the middle row of the current column
        axs[1, idx].imshow(out, cmap='viridis', interpolation='nearest')
        axs[1, idx].set_title(f"Output {idx+1}")
        axs[1, idx].axis('off')

        # Display the predicted category at the bottom row of the current column
        axs[2, idx].text(0.5, 0.5, f"Pred: {predicted_category}",
                         ha='center', va='center', fontsize=12, color='blue', weight='bold')
        axs[2, idx].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the title to avoid overlap
    plt.show()

# === Main classification function ===
def classify_arc_task(task_data, task_name="ARC Task"):
    """
    Classifies an ARC task based on its input-output transformation.
    It evaluates the rules for each category and calculates a Solomonoff score to predict the most likely category.
    """
    train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task_data['train']]
    scores = {}
    print("\n=== Evaluating Task ===")
    for category, rules in CATEGORY_RULES.items():
        print(f"\n--- Category: {category} ---")
        score = solomonoff_score(rules, train_pairs, category)
        scores[category] = score
        print(f"Total Solomonoff Score for {category}: {score:.5e}")
    best = max(scores, key=scores.get)
    print(f"\nBest category prediction: {best}\n")

    # Show visualisation for current task with prediction
    visualise_all_train_pairs_with_prediction(task_data, task_name, predicted_category=best)

    return best, scores

# === Load ARC task ===
def load_arc_task(path):
    """
    Loads an ARC task from a JSON file and returns the data.
    """
    with open(path, 'r') as f:
        return json.load(f)

# === Batch classification with improved logging ===
def classify_folder(folder):
    """
    Classifies all ARC tasks in a given folder and logs the results.
    """
    folder_path = Path(folder)
    results = []
    for json_file in folder_path.glob("*.json"):
        print(f"\nClassifying task: {json_file.name}")
        task_data = load_arc_task(json_file)
        task_name = json_file.stem
        prediction, scores = classify_arc_task(task_data, task_name)
        results.append((task_name, prediction, scores))
    return results


# === __main__ ===
if __name__ == "__main__":
    arc_folder = "./MINI-ARC/data/MiniARC" # MiniARC
    #arc_folder = "./Training" # Standard ARC
    results = classify_folder(arc_folder)




### JOSH NOTES...

# How it works:

# Loops through each category (Colour, Commonsense, etc)

# Within each category, it evaluates rules (functions defining what makes those inherent logics)

# It applies the rules to every pair, if it passes for all pairs it counts as 'valid'

# The score is a solomonoff prior, where the least complex solutions are desireable 
# adds up all priors of all passing rules in the category to get the total category score
# the category with the highest score is the best match

# Solomonoff Prior is: P(h) = s^(-complexity(h))
# The higher the score, the more probable the category is.


## MATH

# 1. **Rule Complexity:**
#    - Each rule is measured by its bytecode length.
#    - The length of the bytecode is a rough measure of the rule's complexity.

# 2. **Solomonoff Prior:**
#    - The prior probability of a rule is given by: 
#      \[ P(h) = 2^{-\text{complexity}(h)} \]
#    - A simpler rule (lower complexity) has a higher prior (more likely), and a more complex rule (higher complexity) has a lower prior.

# 3. **Rule Evaluation:**
#    - Each rule is applied to all training pairs. If the rule successfully applies to all pairs, it contributes to the score.
#    - The total score for a category is the sum of the priors of all successful rules.

# 4. **Category Score:**
#    - The category with the highest total score is selected as the most likely category for the task.
