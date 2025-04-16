import matplotlib.pyplot as plt
import numpy as np

def visualise_pair_with_prediction(inp, out, predicted_category, pair_idx, cmap="viridis"):
    """
    Visualises a single input-output pair with its predicted category.

    Parameters:
        inp (ndarray): 2D NumPy array for input grid.
        out (ndarray): 2D NumPy array for output grid.
        predicted_category (str): Predicted category label.
        pair_idx (int): Index of the current pair.
        cmap (str): Colormap to use.
    """
    # Create a 3-row subplot layout (Input, Output, Prediction)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows for input, output, prediction
    
    # Display Input grid
    axs[0].imshow(inp, cmap=cmap)
    axs[0].set_title(f"Input {pair_idx+1}")
    axs[0].axis('off')

    # Display Output grid
    axs[1].imshow(out, cmap=cmap)
    axs[1].set_title(f"Output {pair_idx+1}")
    axs[1].axis('off')

    # Display Predicted Category
    axs[2].text(0.5, 0.5, f"Predicted: {predicted_category}",
                ha='center', va='center', fontsize=14, color='blue', weight='bold')
    axs[2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the title to avoid overlap
    plt.show()


def plot_grids(inp, out, title="Input vs Output", cmap="tab20", show=True):
    """
    Displays input and output grids side-by-side using matplotlib.
    
    Parameters:
        inp (ndarray): 2D NumPy array for input grid.
        out (ndarray): 2D NumPy array for output grid.
        title (str): Title of the entire plot.
        cmap (str): Colormap to use.
        show (bool): Whether to call plt.show().
    """
    # Use the new structure for visualizing each pair
    visualise_pair_with_prediction(inp, out, "N/A", pair_idx=0, cmap=cmap)
    if show:
        plt.show()


def display_rule_results(rule_results, rule_names, pair_idx=None):
    """
    Prints which rules passed or failed for a given input/output pair.
    
    Parameters:
        rule_results (List[bool]): List of True/False values indicating rule success.
        rule_names (List[str]): Corresponding names of rules.
        pair_idx (int or None): Optional index of the input/output pair.
    """
    header = f"Results for pair {pair_idx}" if pair_idx is not None else "Rule Results"
    print(header)
    print("-" * len(header))
    for rule, passed in zip(rule_names, rule_results):
        status = "✅" if passed else "❌"
        print(f"{rule:30s}: {status}")
    print()


def plot_solomonoff_scores(score_dict):
    """
    Plots a bar chart of Solomonoff scores for each rule category.

    Parameters:
        score_dict (dict): Keys are category names, values are Solomonoff scores (float).
    """
    categories = list(score_dict.keys())
    scores = [score_dict[cat] for cat in categories]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(categories, scores, color='mediumslateblue')
    plt.ylabel('Solomonoff Score')
    plt.title('Rule Category Complexity (Solomonoff Scores)')
    plt.xticks(rotation=30, ha='right')
    
    # Annotate bars with score values
    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def compare_multiple_pairs(pairs, task_id="Unknown Task", cmap="tab20", predicted_categories=None):
    """
    Visualizes multiple (input, output) grid pairs in a column, with predictions displayed below each column.

    Parameters:
        pairs (List[Tuple[np.ndarray, np.ndarray]]): List of (input, output) pairs.
        task_id (str): Optional task ID to show as title.
        cmap (str): Matplotlib colormap to use.
        predicted_categories (List[str] or None): Optional predicted categories for each pair.
    """
    num_pairs = len(pairs)
    
    # Create a 3-row layout: Input, Output, and Predictions
    fig, axes = plt.subplots(3, num_pairs, figsize=(8 * num_pairs, 6))  # 3 rows: input, output, prediction

    if num_pairs == 1:
        axes = [axes]  # ensure iterable

    for idx, (inp, out) in enumerate(pairs):
        ax_in, ax_out, ax_pred = axes[:, idx]  # Accessing three rows for each pair

        # Display Input grid
        ax_in.imshow(inp, cmap=cmap)
        ax_in.set_title(f"Pair {idx + 1} - Input")
        ax_in.axis('off')

        # Display Output grid
        ax_out.imshow(out, cmap=cmap)
        ax_out.set_title(f"Pair {idx + 1} - Output")
        ax_out.axis('off')

        # Display Predicted Category
        if predicted_categories:
            ax_pred.text(0.5, 0.5, f"Pred: {predicted_categories[idx]}", ha='center', va='center', color='black', fontsize=12)
        ax_pred.axis('off')  # Hide axes for the prediction row

    fig.suptitle(f"Task {task_id} - Input/Output Examples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the title to avoid overlap
    plt.show()


