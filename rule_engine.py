from tqdm import tqdm
import numpy as np
from pathlib import Path
from rules import ALL_RULES
from solomonoff import calculate_solomonoff_score
from categories import CATEGORIES
from utils.complexity import rule_complexity
from utils.visualisation import *
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dsl import *

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RuleEngine:
    def __init__(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.task_data = {}
        self.category_scores = {}

    def manual_categorize(self):
        import matplotlib.pyplot as plt

        self.output_folder.mkdir(parents=True, exist_ok=True)
        manual_path = Path("manual_categorization.json")

        tasks = list(self.data_folder.glob("*.json"))
        manual_results = {}

        print("\nManual Categorization Mode")
        print("Categories:")
        for i, cat in enumerate(CATEGORIES):
            print(f"{i}: {cat}")

        plt.ion()  # Turn on interactive mode

        for idx, task_path in enumerate(tasks):
            task_name = task_path.name
            with open(task_path) as f:
                task = json.load(f)

            pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]

            # Visualize the task
            fig = compare_multiple_pairs(pairs, task_id=task_name)
            plt.pause(0.001)  # Show non-blocking plot

            # Prompt for input
            while True:
                try:
                    inp = input(f"\nTask {idx+1}/{len(tasks)}: {task_name}\nEnter category number (or 's' to skip): ").strip()
                    if inp.lower() == 's':
                        print(f"⏭️ Skipped {task_name}")
                        break
                    elif inp.isdigit() and 0 <= int(inp) < len(CATEGORIES):
                        manual_results[task_name] = CATEGORIES[int(inp)]
                        print(f"✔️ Saved: {task_name} → {CATEGORIES[int(inp)]}")
                        break
                    else:
                        print(f"Invalid input. Please enter a number between 0 and {len(CATEGORIES)-1}, or 's' to skip.")
                except KeyboardInterrupt:
                    print("\nExiting manual categorization.")
                    plt.close("all")
                    return

            plt.close("all")  # Close after each entry

        with open(manual_path, "w") as f:
            json.dump(manual_results, f, indent=2)

        print(f"\n✅ Manual categorization saved to: {manual_path}")


    def run(self, save_results=True):
        self.output_folder.mkdir(parents=True, exist_ok=True)

        manual_path = Path("manual_categorization.json")
        manual_results = {}
        if manual_path.exists():
            with open(manual_path) as f:
                manual_results = json.load(f)

        scores_path = self.output_folder / "evaluated_scores.json"
        if scores_path.exists():
            scores_path.unlink()
            print(f"🗑️ Deleted existing {scores_path.name}")

        tasks = list(self.data_folder.glob("*.json"))
        total_tasks = len(tasks)
        all_results = {}

        correct_count = 0
        correct_tasks = []  # <- New line to store correct task names

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

            scores = np.array(list(category_scores.values()))
            normalized_scores = softmax(scores)

            normalized_category_scores = {
                category: normalized_scores[idx] for idx, category in enumerate(CATEGORIES)
            }

            best_category = max(normalized_category_scores, key=normalized_category_scores.get)
            task['predicted_scores'] = normalized_category_scores
            task['predicted_categories'] = [best_category] * len(task["train"])
            if manual_results and task_name in manual_results:
                task['expected_category'] = manual_results[task_name]

            if save_results:
                output_path = self.output_folder / f"{task_path.stem}_evaluated.json"
                with open(output_path, "w") as out_f:
                    json.dump(task, out_f, indent=2)

            expected_category = manual_results.get(task_name) if manual_results else None
            all_results[task_name] = {
                "predicted_category": best_category,
                "expected_category": expected_category,
                "scores": normalized_category_scores
            }


            # Compare with manual if available
            if manual_results:
                manual_cat = manual_results.get(task_name)
                if manual_cat == best_category:
                    correct_count += 1
                    correct_tasks.append((idx, task_name))  # <- Track task number and name
                    #print(f"✅ Correctly matched [#{idx}] {task_name} → {best_category}")

        print("Evaluation complete. Results saved.")

        if save_results:
            with open(scores_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Scores saved to: {scores_path.resolve()}")

        if manual_results:
            accuracy = correct_count / len(manual_results) * 100
            print(f"\n🤖 AI vs 👤 Human categorization accuracy: {accuracy:.2f}% ({correct_count}/{len(manual_results)})")
            print("\nCorrectly matched tasks:")
            for idx, task in correct_tasks:
                print(f" - [#{idx}] {task}")

        print("Evaluation complete. Results saved.")

        if save_results:
            with open(scores_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Scores saved to: {scores_path.resolve()}")

        if manual_results:
            accuracy = correct_count / len(manual_results) * 100
            print(f"\n🤖 AI vs 👤 Human categorization accuracy: {accuracy:.2f}% ({correct_count}/{len(manual_results)})")

    def evaluate_category(self, task, category, rules):
        if "train" not in task or not task["train"]:
            print(f"❌ Missing or empty 'train' in task: {task}")
            return 0.0

        first_pair = task["train"][0]
        inp_grid = np.array(first_pair["input"])
        out_grid = np.array(first_pair["output"])

        inp_objs = objects(tuple(tuple(row) for row in first_pair["input"]), True, True, True)
        out_objs = objects(tuple(tuple(row) for row in first_pair["output"]), True, True, True)

        total_score = 0.0

        for rule_func, prior in rules:
            try:
                passed = rule_func(inp_grid, out_grid, inp_objs, out_objs)
            except TypeError:
                passed = rule_func(inp_grid, out_grid)

            if passed:
                complexity = rule_complexity(rule_func)
                score = calculate_solomonoff_score([True], prior, complexity)
                total_score += score

        return total_score

    def View(self, task_name=None):
        if not self.task_data:
            print("No tasks loaded. Run the engine first.")
            return

        task_names = list(self.task_data.keys())

        # Handle integer index input
        if isinstance(task_name, int):
            if 0 <= task_name < len(task_names):
                task_names = [task_names[task_name]]
            else:
                print(f"Invalid index: {task_name}. Must be between 0 and {len(task_names)-1}.")
                return
        elif isinstance(task_name, str):
            task_names = [task_name]

        for task_name in task_names:
            print(f"\n--- Visualising Task: {task_name} ---")

            evaluated_path = self.output_folder / task_name.replace(".json", "_evaluated.json")
            if evaluated_path.exists():
                with open(evaluated_path) as f:
                    task = json.load(f)
            else:
                task = self.task_data.get(task_name)

            if task is None:
                print(f"Task '{task_name}' not found.")
                continue

            try:
                scores_path = self.output_folder / "evaluated_scores.json"
                with open(scores_path, "r") as f:
                    self.category_scores = json.load(f)
            except FileNotFoundError:
                print("Evaluated scores file not found. Please run the engine first.")
                return

            pairs = [(np.array(pair["input"]), np.array(pair["output"])) for pair in task["train"]]
            predicted_categories = task.get("predicted_categories", [])
            expected_category = task.get("expected_category", None)
            compare_multiple_pairs(
                pairs, 
                task_id=task_name, 
                predicted_categories=predicted_categories,
                expected_category=expected_category
            )

            for category in CATEGORIES:
                rules = ALL_RULES.get(category, [])
                if not rules:
                    continue
                rule_names = [func.__name__ for func, _ in rules]
                pair = task["train"][0]
                inp = np.array(pair["input"])
                out = np.array(pair["output"])
                results = []
                for func, _ in rules:
                    try:
                        inp_objs = objects(tuple(tuple(row) for row in pair["input"]), True, True, True)
                        out_objs = objects(tuple(tuple(row) for row in pair["output"]), True, True, True)
                        results.append(func(inp, out, inp_objs, out_objs))
                    except TypeError:
                        results.append(func(inp, out))
                display_rule_results(results, rule_names)

            score_dict = self.category_scores.get(task_name, {})
            if not score_dict:
                print("No score data found for this task.")
                continue

            plot_solomonoff_scores(score_dict)
            self.plot_task_objects(task_name)
            plt.show()

    def plot_task_objects(self, task_name):
        if task_name not in self.task_data:
            print(f"Task '{task_name}' not found in loaded data.")
            return

        task = self.task_data[task_name]
        train_pairs = task.get("train", [])

        for idx, pair in enumerate(train_pairs):
            for mode in ["input", "output"]:
                grid = tuple(tuple(row) for row in pair[mode])
                objs = objects(grid=grid, univalued=True, diagonal=True, without_bg=True)
                fig, ax = plt.subplots()
                ax.imshow(grid, cmap="tab20", interpolation="none")

                for obj in objs:
                    indices = [loc for _, loc in obj]
                    rows, cols = zip(*indices)
                    min_row, max_row = min(rows), max(rows)
                    min_col, max_col = min(cols), max(cols)
                    rect = patches.Rectangle(
                        (min_col - 0.5, min_row - 0.5),
                        max_col - min_col + 1,
                        max_row - min_row + 1,
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                ax.set_title(f"Task: {task_name} | Pair #{idx} | {mode.capitalize()}")
                plt.axis("off")
