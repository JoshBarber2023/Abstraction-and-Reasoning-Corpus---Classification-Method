import numpy as np
from pathlib import Path
from rules import ALL_RULES
from solomonoff import calculate_solomonoff_score
from categories import CATEGORIES
from utils.complexity import rule_complexity
from utils.visualisation import plot_grids, display_rule_results, plot_solomonoff_scores, compare_multiple_pairs
import json

class RuleEngine:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.task_data = {}

    def run(self):
        tasks = list(self.data_folder.glob("*.json"))
        for task_path in tasks:
            with open(task_path) as f:
                task = json.load(f)

            self.task_data[task_path.name] = task

            #print(f"\n=== Evaluating Task: {task_path.name} ===")
            for category in CATEGORIES:
                category_rules = ALL_RULES.get(category, [])
                score = self.evaluate_category(task, category, category_rules)
                #print(f"Solomonoff Score [{category}]: {score:.4e}")

    def evaluate_category(self, task, category, rules):
        total_score = 0.0
        for rule_func, prior in rules:
            complexity = rule_complexity(rule_func)
            passed_results = []
            for pair in task["train"]:
                inp, out = np.array(pair["input"]), np.array(pair["output"])
                passed_results.append(rule_func(inp, out))
            rule_score = calculate_solomonoff_score(passed_results, prior, complexity)
            total_score += rule_score
            #print(f"- {rule_func.__name__}: passed={all(passed_results)}, complexity={complexity}, prior={prior:.2f}, passed_per_pair={passed_results}")
        return total_score
    
    def View(self, task_name=None):
        """
        Visualizes the grids and rule evaluations for a specific task.

        Parameters:
            task_name (str or None): Filename of the task to view. If None, view the first task.
        """
        if not self.task_data:
            print("No tasks loaded. Run the engine first.")
            return

        if task_name is None:
            task_name = list(self.task_data.keys())[0]

        task = self.task_data.get(task_name)
        if task is None:
            print(f"Task '{task_name}' not found.")
            return

        print(f"\n--- Visualizing Task: {task_name} ---")

        # Show all train input/output pairs
        pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
        
        # Default to an empty list if 'predicted_categories' is not available
        predicted_categories = task.get("predicted_categories", [])

        # Pass predicted_categories to the compare_multiple_pairs function
        compare_multiple_pairs(pairs, task_id=task_name, predicted_categories=predicted_categories)

        # Show rule results for first train pair for each category
        for category in CATEGORIES:
            rules = ALL_RULES.get(category, [])
            if not rules:
                continue
            rule_names = [func.__name__ for func, _ in rules]
            pair = task["train"][0]
            inp, out = np.array(pair["input"]), np.array(pair["output"])
            results = [func(inp, out) for func, _ in rules]
            display_rule_results(results, rule_names)

        # Show Solomonoff scores for this task
        score_dict = {}
        for category in CATEGORIES:
            rules = ALL_RULES.get(category, [])
            score = self.evaluate_category(task, category, rules)
            score_dict[category] = score
        plot_solomonoff_scores(score_dict)


