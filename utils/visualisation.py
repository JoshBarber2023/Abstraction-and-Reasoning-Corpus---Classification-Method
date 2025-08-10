import matplotlib.pyplot as plt
import numpy as np

def visualise_pair_with_prediction(inp, out, predicted_category, pair_idx, cmap="viridis"):
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
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

def display_rule_results(rule_results, pair_idx=None):
    """
    rule_results: dict {category: {rule_description: bool, ...}, ...}
    """
    header = f"Passing Rules for pair {pair_idx}" if pair_idx is not None else "Passing Rules"
    print(header)
    print("-" * len(header))
    found_any = False
    for category, rules in rule_results.items():
        passing_rules = [desc for desc, passed in rules.items() if passed]
        if passing_rules:
            found_any = True
            print(f"\nCategory: {category}")
            for rule_desc in passing_rules:
                print(f" - ✅ {rule_desc}")
    if not found_any:
        print("No passing rules found.")
    print()

def display_detailed_hypotheses(detailed_hypotheses, version=""):
    """
    detailed_hypotheses: dict {category: list of [desc, bool, ...]}
    """
    print(f"Detailed hypotheses {version}:")
    for category, hypotheses in detailed_hypotheses.items():
        print(f"\nCategory: {category}")
        for hypothesis in hypotheses:
            desc, passed = hypothesis[0], hypothesis[1]
            status = "✅" if passed else "❌"
            print(f" - {status} {desc}")

def plot_solomonoff_scores(scores_dict, title="Solomonoff Scores"):
    """
    scores_dict: {category: score, ...}
    """
    categories = list(scores_dict.keys())
    scores = [scores_dict[cat] for cat in categories]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(categories, scores, color='mediumslateblue')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(rotation=10, ha='right')

    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{score:.3f}', ha='center', va='bottom')

def compare_multiple_pairs(pairs, task_id="Unknown Task", cmap="tab20", predicted_categories=None, expected_category=None):
    num_pairs = len(pairs)
    fig, axes = plt.subplots(3, num_pairs, figsize=(6 * num_pairs, 8))

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

        if predicted_categories and idx < len(predicted_categories):
            pred_text = f"Predicted: {predicted_categories[idx]}"
            if expected_category:
                pred_text += f"\nExpected: {expected_category}"
            ax_pred.text(0.5, 0.5, pred_text,
                         ha='center', va='center', color='black', fontsize=12)

        ax_pred.axis('off')

    fig.suptitle(f"Task {task_id} - Input/Output Examples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)
