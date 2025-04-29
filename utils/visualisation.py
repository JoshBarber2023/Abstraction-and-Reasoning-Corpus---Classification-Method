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
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows for input, output, prediction
    
    axs[0].imshow(inp, cmap=cmap)
    axs[0].set_title(f"Input {pair_idx + 1}")
    axs[0].axis('off')

    axs[1].imshow(out, cmap=cmap)
    axs[1].set_title(f"Output {pair_idx + 1}")
    axs[1].axis('off')

    axs[2].text(0.5, 0.5, f"Predicted: {predicted_category}",
                ha='center', va='center', fontsize=14, color='blue', weight='bold')
    axs[2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)


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
    # Extract the 'scores' dictionary
    scores_dict = score_dict['scores']
    
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
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{score:.2e}', ha='center', va='bottom')

def compare_multiple_pairs(pairs, task_id="Unknown Task", cmap="tab20", predicted_categories=None):
    """
    Visualizes multiple (input, output) grid pairs with optional predictions.

    Parameters:
        pairs (List[Tuple[np.ndarray, np.ndarray]]): (input, output) pairs.
        task_id (str): Task ID to display as title.
        cmap (str): Colormap for imshow.
        predicted_categories (List[str] or None): Predicted labels per pair.
    """
    num_pairs = len(pairs)
    fig, axes = plt.subplots(3, num_pairs, figsize=(6 * num_pairs, 8))

    # Ensure 2D axes even if num_pairs == 1
    if num_pairs == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, (inp, out) in enumerate(pairs):
        ax_in, ax_out, ax_pred = axes[:, idx]

        ax_in.imshow(inp, cmap=cmap)
        ax_in.set_title(f"Pair {idx + 1} - Input")
        ax_in.axis('off')

        ax_out.imshow(out, cmap=cmap)
        ax_out.set_title(f"Pair {idx + 1} - Output")
        ax_out.axis('off')

        if predicted_categories:
            ax_pred.text(0.5, 0.5, f"Pred: {predicted_categories[idx]}", 
                         ha='center', va='center', color='black', fontsize=12)
        ax_pred.axis('off')

    fig.suptitle(f"Task {task_id} - Input/Output Examples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
