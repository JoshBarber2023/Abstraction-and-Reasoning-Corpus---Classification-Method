import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import textwrap
import time

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
    
    axs[0].imshow(inp, cmap=cmap, interpolation='nearest')
    axs[0].set_title(f"Input {pair_idx + 1}")
    axs[0].axis('off')
    
    # Add object numbers to input
    if show_object_numbers:
        add_object_numbers_to_plot(axs[0], inp)

    axs[1].imshow(out, cmap=cmap, interpolation='nearest')
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
    try:
        if grid.size == 0:
            return
            
        height, width = grid.shape
        
        # Skip if grid is too large to prevent overcrowding
        if height > 30 or width > 30:
            return
        
        for i in range(height):
            for j in range(width):
                try:
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
                           fontsize=max(6, min(10, 200 // max(height, width))), 
                           fontweight='bold',
                           color=text_color)
                except:
                    continue  # Skip problematic cells
    except Exception as e:
        print(f"Warning: Could not add object numbers: {e}")


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
    
    # Limit figure width to prevent memory issues
    fig_width = min(6 * num_pairs, 24)
    fig, axes = plt.subplots(3, num_pairs, figsize=(fig_width, 8))

    # Ensure 2D axes even if num_pairs == 1
    if num_pairs == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, (inp, out) in enumerate(pairs):
        ax_in, ax_out, ax_pred = axes[:, idx]

        # Display input grid
        ax_in.imshow(inp, cmap=cmap, interpolation='nearest')
        ax_in.set_title(f"Pair {idx + 1} - Input")
        ax_in.axis('off')
        
        # Add object numbers to input
        if show_object_numbers:
            add_object_numbers_to_plot(ax_in, inp)

        # Display output grid
        ax_out.imshow(out, cmap=cmap, interpolation='nearest')
        ax_out.set_title(f"Pair {idx + 1} - Output")
        ax_out.axis('off')
        
        # Add object numbers to output
        if show_object_numbers:
            add_object_numbers_to_plot(ax_out, out)

        # Display prediction info
        if predicted_categories and idx < len(predicted_categories):
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


def visualize_task_comprehensive(task_name, task_data, interpretation_data=None, 
                               scores_data=None, hypotheses_data=None, 
                               cmap="tab20", show_object_numbers=True, 
                               save_path=None):
    """
    Comprehensive visualization of a single task with all analysis results.
    Improved layout with separate analysis window for complete visibility.
    """
    print(f"🎨 Creating visualization for: {task_name}")
    
    try:
        # Extract training pairs with validation
        pairs = []
        train_data = task_data.get("train", [])
        
        if not train_data:
            print(f"⚠️ No training pairs found for {task_name}")
            return None
        
        for pair in train_data:
            try:
                inp = np.array(pair["input"])
                out = np.array(pair["output"])
                pairs.append((inp, out))
            except Exception as e:
                print(f"Warning: Could not load pair: {e}")
                continue
        
        if not pairs:
            print(f"❌ No valid pairs loaded for {task_name}")
            return None
        
        num_pairs = len(pairs)
        print(f"   📊 Processing {num_pairs} pairs")
        
        # Create figure with improved layout - more space for analysis
        fig_width = min(max(14, 4 * num_pairs), 24)
        fig_height = 16  # Increased height for better text visibility
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Improved grid layout with more space for analysis
        gs = fig.add_gridspec(5, num_pairs, 
                             height_ratios=[1, 1, 0.15, 0.8, 0.8], 
                             hspace=0.3, wspace=0.25, 
                             top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # Plot input/output grids
        for idx, (inp, out) in enumerate(pairs):
            try:
                # Input grid
                ax_in = fig.add_subplot(gs[0, idx])
                ax_in.imshow(inp, cmap=cmap, interpolation='nearest')
                ax_in.set_title(f"Input {idx + 1}", fontsize=11, fontweight='bold')
                ax_in.axis('off')
                
                if show_object_numbers:
                    add_object_numbers_to_plot(ax_in, inp)
                
                # Output grid
                ax_out = fig.add_subplot(gs[1, idx])
                ax_out.imshow(out, cmap=cmap, interpolation='nearest')
                ax_out.set_title(f"Output {idx + 1}", fontsize=11, fontweight='bold')
                ax_out.axis('off')
                
                if show_object_numbers:
                    add_object_numbers_to_plot(ax_out, out)
                    
            except Exception as e:
                print(f"Warning: Could not plot pair {idx+1}: {e}")
                continue
        
        # Prediction info row
        try:
            ax_pred = fig.add_subplot(gs[2, :])
            ax_pred.axis('off')
            
            predicted_category = "Unknown"
            if task_data.get("predicted_categories"):
                if isinstance(task_data["predicted_categories"], list) and task_data["predicted_categories"]:
                    predicted_category = str(task_data["predicted_categories"][0])
            elif task_data.get("predicted_category"):
                predicted_category = str(task_data["predicted_category"])
                
            expected_category = task_data.get("expected_category")
            
            pred_text = f"Predicted: {predicted_category}"
            if expected_category:
                match_symbol = "✅ CORRECT" if predicted_category == expected_category else "❌ INCORRECT"
                pred_text += f"  |  Expected: {expected_category} ({match_symbol})"
            
            ax_pred.text(0.5, 0.5, pred_text, ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='darkblue',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
                        
        except Exception as e:
            print(f"Warning: Could not add prediction info: {e}")
        
        # Analysis panels - split into two rows for better visibility
        try:
            # Interpretation panel
            ax_interp = fig.add_subplot(gs[3, :])
            ax_interp.axis('off')
            
            interpretation_text = create_interpretation_text(interpretation_data)
            
            ax_interp.text(0.02, 0.98, interpretation_text, transform=ax_interp.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
                          linespacing=1.3)
            
            # Scores and Hypotheses panel
            ax_analysis = fig.add_subplot(gs[4, :])
            ax_analysis.axis('off')
            
            analysis_text = create_scores_hypotheses_text(scores_data, hypotheses_data)
            
            ax_analysis.text(0.02, 0.98, analysis_text, transform=ax_analysis.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
                            linespacing=1.3)
                            
        except Exception as e:
            print(f"Warning: Could not add analysis panels: {e}")
        
        # Main title
        fig.suptitle(f"COMPREHENSIVE ANALYSIS: {task_name}", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                           facecolor='white', pad_inches=0.3)
                print(f"💾 Saved: {save_path}")
            except Exception as e:
                print(f"Warning: Could not save image: {e}")
        
        print(f"✅ Successfully created visualization for {task_name}")
        return fig
        
    except Exception as e:
        print(f"❌ Error creating visualization for {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_interpretation_text(interpretation_data):
    """Create interpretation text with proper formatting"""
    lines = []
    
    try:
        if interpretation_data and isinstance(interpretation_data, dict):
            lines.append("🧠 INTERPRETATION:")
            lines.append("=" * 50)
            
            # What it does
            what_it_does = str(interpretation_data.get('what_it_does', 'No interpretation available'))
            wrapped_what = textwrap.fill(what_it_does, width=90, initial_indent="", 
                                       subsequent_indent="")
            lines.extend(wrapped_what.split('\n'))
            lines.append("")
            
            # Confidence and success rate
            confidence = interpretation_data.get('confidence', 'Unknown')
            success_rate = interpretation_data.get('success_rate', 'N/A')
            lines.append(f"Confidence: {confidence}")
            lines.append(f"Success Rate: {success_rate}")
            
            # Additional details if available
            if 'pattern_description' in interpretation_data:
                lines.append("")
                lines.append("Pattern Description:")
                pattern_desc = str(interpretation_data['pattern_description'])
                wrapped_pattern = textwrap.fill(pattern_desc, width=90, initial_indent="• ", 
                                              subsequent_indent="  ")
                lines.extend(wrapped_pattern.split('\n'))
                
        else:
            lines.append("🧠 INTERPRETATION: No interpretation data available")
    except Exception as e:
        lines.append(f"⚠️ Error loading interpretation: {e}")
    
    return "\n".join(lines)


def create_scores_hypotheses_text(scores_data, hypotheses_data):
    """Create scores and hypotheses text with proper formatting"""
    lines = []
    
    try:
        # Category scores section
        if scores_data and isinstance(scores_data, dict) and 'scores' in scores_data:
            lines.append("📊 CATEGORY SCORES:")
            lines.append("=" * 25)
            scores = scores_data['scores']
            if isinstance(scores, dict):
                sorted_scores = sorted(scores.items(), key=lambda x: -float(x[1]))
                for i, (cat, score) in enumerate(sorted_scores[:5]):  # Show top 5
                    try:
                        # Create visual bar
                        bar_length = max(1, min(15, int(float(score) * 15)))
                        bar = "█" * bar_length + "░" * (15 - bar_length)
                        lines.append(f"{i+1:2}. {str(cat)[:15]:15} {float(score):.3f} [{bar}]")
                    except:
                        lines.append(f"{i+1:2}. {str(cat)[:15]:15} {score}")
        else:
            lines.append("📊 CATEGORY SCORES: No scores available")
        
        lines.append("")
        lines.append("")
        
        # Hypotheses section
        if hypotheses_data and isinstance(hypotheses_data, list):
            lines.append("🔬 HYPOTHESES ANALYSIS:")
            lines.append("=" * 30)
            
            total_hyps = len(hypotheses_data)
            passed_all = sum(1 for h in hypotheses_data if isinstance(h, dict) and h.get('passed_all', False))
            
            lines.append(f"Total Hypotheses: {total_hyps}")
            lines.append(f"Perfect Matches: {passed_all}")
            lines.append("")
            
            # Show top hypotheses
            valid_hyps = [h for h in hypotheses_data if isinstance(h, dict)]
            if valid_hyps:
                # Sort by pass rate
                sorted_hyps = sorted(valid_hyps, key=lambda x: x.get('pass_rate', 0), reverse=True)
                
                lines.append("Top Hypotheses:")
                for i, hyp in enumerate(sorted_hyps[:3]):  # Show top 3
                    try:
                        status_icon = "✅" if hyp.get('passed_all', False) else "❌"
                        pass_rate = hyp.get('pass_rate', 0)
                        category = str(hyp.get('category', 'Unknown'))[:12]
                        description = str(hyp.get('description', 'No description'))
                        
                        # Wrap long descriptions
                        if len(description) > 60:
                            description = description[:57] + "..."
                        
                        lines.append(f"{i+1}. {status_icon} [{category:12}] {pass_rate:5.1%}")
                        lines.append(f"   {description}")
                        
                        if i < len(sorted_hyps) - 1:  # Add space between entries
                            lines.append("")
                            
                    except Exception as e:
                        lines.append(f"   Error processing hypothesis {i+1}: {e}")
            else:
                lines.append("No valid hypotheses found")
        else:
            lines.append("🔬 HYPOTHESES: No hypotheses data available")
            
    except Exception as e:
        lines.append(f"⚠️ Error creating analysis text: {e}")
    
    return "\n".join(lines)


def visualize_all_tasks_comprehensive(engine, max_tasks=None, save_individual=False, 
                                    auto_advance=False, save_dir=None):
    """
    Visualize all tasks with comprehensive analysis in an interactive slideshow format.
    
    Parameters:
        engine: The HybridRuleEngine instance
        max_tasks (int): Maximum number of tasks to visualize
        save_individual (bool): Whether to save individual task images
        auto_advance (bool): Whether to auto-advance through tasks
        save_dir (str): Directory to save images
    """
    if not hasattr(engine, 'task_data') or not engine.task_data:
        print("❌ No tasks loaded. Run load_tasks() first.")
        return
    
    # Load all necessary data with error handling
    scores_path = engine.output_folder / "evaluated_scores.json"
    interpretations_path = engine.output_folder / "interpretations_summary.json"
    hypotheses_path = engine.output_folder / "generated_hypotheses.json"
    
    all_scores = {}
    all_interpretations = {}
    all_hypotheses = {}
    
    try:
        if scores_path.exists():
            with open(scores_path) as f:
                all_scores = json.load(f)
                print(f"✅ Loaded scores for {len(all_scores)} tasks")
    except Exception as e:
        print(f"⚠️ Warning: Could not load scores: {e}")
        
    try:
        if interpretations_path.exists():
            with open(interpretations_path) as f:
                interp_data = json.load(f)
                all_interpretations = interp_data.get("problems", {})
                print(f"✅ Loaded interpretations for {len(all_interpretations)} tasks")
    except Exception as e:
        print(f"⚠️ Warning: Could not load interpretations: {e}")
        
    try:
        if hypotheses_path.exists():
            with open(hypotheses_path) as f:
                all_hypotheses = json.load(f)
                print(f"✅ Loaded hypotheses for {len(all_hypotheses)} tasks")
    except Exception as e:
        print(f"⚠️ Warning: Could not load hypotheses: {e}")
    
    # Get tasks to visualize
    task_names = list(engine.task_data.keys())
    if max_tasks:
        task_names = task_names[:max_tasks]
    
    print(f"\n🎬 COMPREHENSIVE VISUALIZATION MODE")
    print(f"📊 Showing {len(task_names)} tasks with full analysis")
    print("=" * 60)
    
    if save_individual and save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving images to: {save_path}")
    
    # Interactive visualization loop
    current_idx = 0
    
    while current_idx < len(task_names):
        task_name = task_names[current_idx]
        
        print(f"\n🔍 Task {current_idx + 1}/{len(task_names)}: {task_name}")
        
        try:
            # Load task data
            evaluated_path = engine.output_folder / task_name.replace(".json", "_evaluated.json")
            if evaluated_path.exists():
                with open(evaluated_path) as f:
                    task_data = json.load(f)
            else:
                task_data = engine.task_data.get(task_name, {})
            
            # Get analysis data
            scores_data = all_scores.get(task_name, {})
            interpretation_data = all_interpretations.get(task_name, {})
            hypotheses_data = all_hypotheses.get(task_name, [])
            
            # Create comprehensive visualization
            save_path_current = None
            if save_individual and save_dir:
                save_path_current = Path(save_dir) / f"{task_name.replace('.json', '')}_comprehensive.png"
            
            fig = visualize_task_comprehensive(
                task_name, task_data, interpretation_data,
                scores_data, hypotheses_data, save_path=save_path_current
            )
            
            if fig:
                plt.show()
                
                if not auto_advance:
                    # Interactive navigation
                    print("\n📱 Navigation:")
                    print("  Enter/Space: Next task")
                    print("  'b': Previous task")
                    print("  'j <num>': Jump to task number")
                    print("  's': Save current visualization")
                    print("  'q': Quit")
                    
                    user_input = input("Choice: ").strip().lower()
                    
                    if user_input in ['q', 'quit']:
                        plt.close('all')
                        break
                    elif user_input == 'b' and current_idx > 0:
                        current_idx -= 1
                        plt.close('all')
                        continue
                    elif user_input.startswith('j '):
                        try:
                            jump_to = int(user_input.split()[1]) - 1
                            if 0 <= jump_to < len(task_names):
                                current_idx = jump_to
                                plt.close('all')
                                continue
                        except:
                            pass
                    elif user_input == 's' and fig:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        manual_save_path = f"{task_name.replace('.json', '')}_viz_{timestamp}.png"
                        fig.savefig(manual_save_path, dpi=150, bbox_inches='tight')
                        print(f"💾 Saved: {manual_save_path}")
                
                plt.close('all')
            else:
                print(f"⚠️ Skipping {task_name} due to visualization error")
            
            current_idx += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {task_name}: {e}")
            current_idx += 1
            continue
    
    print(f"\n✅ Visualization complete! Processed {len(task_names)} tasks.")


def softmax(x):
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))  # Subtract max to avoid overflow
    return e_x / e_x.sum(axis=0, keepdims=True)