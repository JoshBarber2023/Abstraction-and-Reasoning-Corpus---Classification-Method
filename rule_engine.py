from tqdm import tqdm
import numpy as np
from pathlib import Path
from rules import ALL_RULES
from solomonoff import calculate_solomonoff_score
from categories import CATEGORIES
from utils.complexity import rule_complexity
from utils.visualisation import display_rule_results, plot_solomonoff_scores, compare_multiple_pairs
import json
import matplotlib.pyplot as plt

class RuleEngine:
    def __init__(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)  # New folder for processed data
        self.task_data = {}
        self.category_scores = {}  # New dictionary to store scores

    def run(self, save_results=True):
        # Ensure output folder exists
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Delete evaluated_scores.json if it exists in output_folder
        scores_path = self.output_folder / "evaluated_scores.json"
        if scores_path.exists():
            scores_path.unlink()
            print(f"🗑️ Deleted existing {scores_path.name}")

        tasks = list(self.data_folder.glob("*.json"))
        total_tasks = len(tasks)

        all_results = {}

        # Wrap the task loop with tqdm for progress tracking
        for idx, task_path in tqdm(enumerate(tasks), total=total_tasks, desc="Processing tasks", unit="task"):
            with open(task_path) as f:
                task = json.load(f)

            task_name = task_path.name
            self.task_data[task_name] = task

            category_scores = {}
            for category in CATEGORIES:
                category_rules = ALL_RULES.get(category, [])
                score = self.evaluate_category(task, category, category_rules)
                category_scores[category] = score

            # Store scores and best category
            best_category = max(category_scores, key=category_scores.get)
            task['predicted_scores'] = category_scores
            task['predicted_categories'] = [best_category] * len(task["train"])  # Same prediction for all pairs

            # Save updated task JSON to the output folder
            if save_results:
                output_path = self.output_folder / f"{task_path.stem}_evaluated.json"
                with open(output_path, "w") as out_f:
                    json.dump(task, out_f, indent=2)

            all_results[task_name] = {
                "predicted_category": best_category,
                "scores": category_scores
            }

        print("Evaluation complete. Results saved.")

        # Save the evaluated scores in the output folder
        if save_results:
            with open(scores_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Scores saved to: {scores_path.resolve()}")

    def evaluate_category(self, task, category, rules):
        if "train" not in task:
            print(f"❌ Missing 'train' key in task: {task}")
            return 0.0

        total_score = 0.0
        for rule_func, prior in rules:
            complexity = rule_complexity(rule_func)
            passed_results = []
            for pair in task["train"]:
                inp, out = np.array(pair["input"]), np.array(pair["output"])
                passed_results.append(rule_func(inp, out))
            rule_score = calculate_solomonoff_score(passed_results, prior, complexity)
            total_score += rule_score

        # Normalise score by number of rules (if any)
        if len(rules) > 0:
            total_score /= len(rules)

        return total_score

    def View(self, task_name=None):
        if not self.task_data:
            print("No tasks loaded. Run the engine first.")
            return

        # If task_name is provided, only visualise that task, else loop through all tasks
        if task_name:
            task_names = [task_name]
        else:
            task_names = list(self.task_data.keys())  # All tasks if no specific task is given

        # Try loading evaluated version first from the output folder
        for task_name in task_names:
            print(f"\n--- Visualising Task: {task_name} ---")

            # Try to load the task from the evaluated folder
            evaluated_path = self.output_folder / task_name.replace(".json", "_evaluated.json")
            if evaluated_path.exists():
                with open(evaluated_path) as f:
                    task = json.load(f)
            else:
                task = self.task_data.get(task_name)

            if task is None:
                print(f"Task '{task_name}' not found.")
                continue  # Skip this task and move to the next one

            # Try to load scores from the output folder
            try:
                scores_path = self.output_folder / "evaluated_scores.json"
                with open(scores_path, "r") as f:
                    self.category_scores = json.load(f)
            except FileNotFoundError:
                print("Evaluated scores file not found. Please run the engine first.")
                return

            pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
            predicted_categories = task.get("predicted_categories", [])

            # Visualise pairs for this task
            compare_multiple_pairs(pairs, task_id=task_name, predicted_categories=predicted_categories)

            # Visualise rule results for each category
            for category in CATEGORIES:
                rules = ALL_RULES.get(category, [])
                if not rules:
                    continue
                rule_names = [func.__name__ for func, _ in rules]
                pair = task["train"][0]
                inp, out = np.array(pair["input"]), np.array(pair["output"])
                results = [func(inp, out) for func, _ in rules]
                display_rule_results(results, rule_names)

            # Plot the solomonoff scores
            score_dict = self.category_scores.get(task_name, {})
            if not score_dict:
                print("No score data found for this task.")
                continue  # Skip plotting if no score data is found

            plot_solomonoff_scores(score_dict)

            plt.show()
