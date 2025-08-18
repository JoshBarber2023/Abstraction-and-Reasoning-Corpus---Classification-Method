import matplotlib.pyplot as plt
import numpy as np

def visualise_pair_with_prediction(inp, out, predicted_category, pair_idx, cmap="viridis", show_object_numbers=True):
    """
    Visualises a single input-output pair with its predicted category.

    Parameters:
        inp (ndarray): 2D NumPy array for input grid.
        out (ndarray): 2D NumPy array for output grid.
        predicted_category (str): Predicted category label.
        pair_idx (int): Index of the current pair.
        cmap (str): Colormap to use.
        show_object_numbers (bool): Whether to display object numbers on the grids.
    """
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows for input, output, prediction
    
    axs[0].imshow(inp, cmap=cmap)
    axs[0].set_title(f"Input {pair_idx + 1}")
    axs[0].axis('off')
    
    # Add object numbers to input
    if show_object_numbers:
        add_object_numbers_to_plot(axs[0], inp)

    axs[1].imshow(out, cmap=cmap)
    axs[1].set_title(f"Output {pair_idx + 1}")
    axs[1].axis('off')
    
    # Add object numbers to output
    if show_object_numbers:
        add_object_numbers_to_plot(axs[1], out)

    axs[2].text(0.5, 0.5, f"Predicted: {predicted_category}",
                ha='center', va='center', fontsize=14, color='blue', weight='bold')
    axs[2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)


def add_object_numbers_to_plot(ax, grid):
    """
    Add object numbers (unique values) to each cell in the grid plot.
    
    Parameters:
        ax: matplotlib axis to add text to
        grid: 2D numpy array representing the grid
    """
    height, width = grid.shape
    
    for i in range(height):
        for j in range(width):
            value = grid[i, j]
            
            # Choose text color based on background (light vs dark)
            # For better visibility
            if value <= 5:  # Assuming darker colors have lower values
                text_color = 'white'
            else:
                text_color = 'black'
                
            # Add the object number as text
            ax.text(j, i, str(int(value)), 
                   ha='center', va='center', 
                   fontsize=8, fontweight='bold',
                   color=text_color)


def display_rule_results(rule_results, rule_names, pair_idx=None):
    """
    Prints which rules passed or failed for a given input/output pair.

    Parameters:
        rule_results (List[bool]): List of True/False values for rule success.
        rule_names (List[str]): Corresponding names of rules.
        pair_idx (int or None): Index of the input/output pair.
    """
    header = f"Results for pair {pair_idx}" if pair_idx is not None else "Rule Results"
    print(header)
    print("-" * len(header))
    for rule, passed in zip(rule_names, rule_results):
        # Ensure we check if passed is an array or single value
        if isinstance(passed, np.ndarray):  # If passed is an array, evaluate if any element is True
            status = "✅" if passed.any() else "❌"
        else:  # If passed is a single boolean value
            status = "✅" if passed else "❌"
        print(f"{rule:30s}: {status}")
    print()


def plot_solomonoff_scores(score_dict):
    """
    Plots a bar chart of Solomonoff scores per rule category.

    Parameters:
        score_dict (dict): Keys are category names, values are float scores.
    """
    # Extract the 'scores' dictionary if it's nested
    if isinstance(score_dict, dict) and 'scores' in score_dict:
        scores_dict = score_dict['scores']
    else:
        scores_dict = score_dict
    
    # Prepare the data for plotting
    categories = list(scores_dict.keys())
    scores = [scores_dict[cat] for cat in categories]
    
    # Plot the bar chart
    plt.figure(figsize=(10, 5))
    bars = plt.bar(categories, scores, color='mediumslateblue')
    plt.ylabel('Solomonoff Score')
    plt.title('Rule Category Complexity (Solomonoff Scores)')
    plt.xticks(rotation=10, ha='right')
    
    # Annotate each bar with its score
    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.001, f'{score:.3f}', ha='center', va='bottom')


def compare_multiple_pairs(pairs, task_id="Unknown Task", cmap="tab20", predicted_categories=None, expected_category=None, show_object_numbers=True):
    """
    Visualizes multiple (input, output) grid pairs with optional predictions and object numbers.

    Parameters:
        pairs (List[Tuple[np.ndarray, np.ndarray]]): (input, output) pairs.
        task_id (str): Task ID to display as title.
        cmap (str): Colormap for imshow.
        predicted_categories (List[str] or None): Predicted labels per pair.
        expected_category (str or None): Expected category for comparison.
        show_object_numbers (bool): Whether to display object numbers on grids.
    """
    num_pairs = len(pairs)
    fig, axes = plt.subplots(3, num_pairs, figsize=(6 * num_pairs, 8))

    # Ensure 2D axes even if num_pairs == 1
    if num_pairs == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, (inp, out) in enumerate(pairs):
        ax_in, ax_out, ax_pred = axes[:, idx]

        # Display input grid
        ax_in.imshow(inp, cmap=cmap)
        ax_in.set_title(f"Pair {idx + 1} - Input")
        ax_in.axis('off')
        
        # Add object numbers to input
        if show_object_numbers:
            add_object_numbers_to_plot(ax_in, inp)

        # Display output grid
        ax_out.imshow(out, cmap=cmap)
        ax_out.set_title(f"Pair {idx + 1} - Output")
        ax_out.axis('off')
        
        # Add object numbers to output
        if show_object_numbers:
            add_object_numbers_to_plot(ax_out, out)

        # Display prediction info
        if predicted_categories:
            pred_text = f"Pred: {predicted_categories[idx]}"
            if expected_category:
                pred_text += f"\nExpected: {expected_category}"
            ax_pred.text(0.5, 0.5, pred_text, 
                        ha='center', va='center', color='black', fontsize=12)

        ax_pred.axis('off')

    fig.suptitle(f"Task {task_id} - Input/Output Examples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    
def create_object_legend(grid, ax=None):
    """
    Create a legend showing what each object number represents in terms of color.
    
    Parameters:
        grid: 2D numpy array
        ax: matplotlib axis (optional)
    """
    unique_objects = np.unique(grid)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, len(unique_objects) * 0.3))
    
    legend_text = "Object Legend:\n"
    for obj_num in sorted(unique_objects):
        legend_text += f"  {int(obj_num)}: Object {int(obj_num)}\n"
    
    ax.text(0.1, 0.9, legend_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.axis('off')


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max to avoid overflow
    return e_x / e_x.sum(axis=0, keepdims=True)